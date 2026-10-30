import copy
import os
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.action_projection import HYBRID_AP_MODES, project_action_torch
from agents.networks import Critic, CostCritic, make_ap_prior


class DeterministicActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0, ap_prior=None,
                 num_devices=None, num_rsu: int = 1):
        super().__init__()
        self.max_action = float(max_action)
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, action_dim)
        self.ap_prior = make_ap_prior(ap_prior, num_devices, num_rsu) if ap_prior else None
        self.num_devices = num_devices

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        raw = self.out(x)
        if self.ap_prior is not None:
            # v2 hybrid head: tanh for (x, p); per-ED softmax over APs (channel prior added to the
            # logits), mapped to [-1, 1]. A tanh AP block saturates at +1 for several APs, after
            # which argmax ties pick the first AP.
            n2 = 2 * int(self.num_devices)
            logits = raw[:, n2:].reshape(s.shape[0], self.num_devices, -1) + self.ap_prior(s)
            ap = 2.0 * F.softmax(logits, dim=-1) - 1.0
            return torch.cat([torch.tanh(raw[:, :n2]), ap.reshape(s.shape[0], -1)], dim=1) * self.max_action
        return torch.tanh(raw) * self.max_action


@dataclass
class TD3Config:
    gamma: float = 0.99
    tau: float = 0.01
    lr: float = 5e-05
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    exploration_noise: float = 0.1
    exploration_noise_rel_scale: float = 0.5
    exploration_noise_ap_scale: float = 0.2
    policy_noise_rel_scale: float = 0.5
    policy_noise_ap_scale: float = 0.15
    noise_anneal_steps: int = 60000
    noise_anneal_final_scale: float = 0.3


class TD3Agent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        device: str = "cpu",
        cfg: TD3Config = TD3Config(),
        use_lagrangian: bool = False,
        cost_limit: float = 0.22,
        lambda_lr: float = 0.0005,
        lambda_mode: str = "adaptive",
        fixed_lambda: float = 1.0,
        initial_lambda: float = 1.0,
        slow_lambda_factor: float = 0.2,
        cost_coef: float = 1.0,
        update_lambda_in_batch: bool = False,
        cost_critic_target_mode: str = "immediate",
        num_devices=None,
        num_rsu: int = 1,
        ap_projection_mode: str = "legacy",
        ap_prior=None,
    ):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.max_action = float(max_action)
        self.num_devices = int(num_devices) if num_devices is not None else int(action_dim // 3)
        self.num_rsu = int(num_rsu)
        self.ap_projection_mode = str(ap_projection_mode).lower()
        self.device = torch.device(device)
        self.cfg = cfg
        self.alpha = 0.0
        self.use_lagrangian = bool(use_lagrangian)
        self.cost_limit = float(cost_limit)
        self.lambda_lr = float(lambda_lr)
        self.lambda_mode = str(lambda_mode).lower()
        self.fixed_lambda = float(fixed_lambda)
        self.initial_lambda = float(initial_lambda)
        self.slow_lambda_factor = float(slow_lambda_factor)
        self.cost_coef = float(cost_coef)
        self.update_lambda_in_batch = bool(update_lambda_in_batch)
        self.cost_critic_target_mode = str(cost_critic_target_mode).lower()
        self.max_lambda = 50.0
        self.min_lambda = 0.01
        valid_lambda_modes = {"adaptive", "fixed", "slow"}
        if self.lambda_mode not in valid_lambda_modes:
            raise ValueError(
                f"Unknown lambda_mode={self.lambda_mode}, expected one of {valid_lambda_modes}"
            )
        valid_cost_target_modes = {"immediate", "discounted_average", "discounted_return"}
        if self.cost_critic_target_mode not in valid_cost_target_modes:
            raise ValueError(
                f"Unknown cost_critic_target_mode={self.cost_critic_target_mode}, expected one of {valid_cost_target_modes}"
            )
        init_lambda = self.fixed_lambda if self.lambda_mode == "fixed" else self.initial_lambda
        init_lambda = max(self.min_lambda, min(self.max_lambda, init_lambda))
        self.lambda_val = float(init_lambda)
        self.diag_enabled = self.use_lagrangian and os.getenv("TD3_LAG_DIAG", "0").lower() not in {
            "",
            "0",
            "false",
            "off",
        }
        yc_clamp_max_raw = os.getenv("TD3_LAG_YC_CLAMP_MAX", "2.5").strip().lower()
        if yc_clamp_max_raw in {"none", "off", "inf", "infinity"}:
            self.yc_clamp_max = None
        else:
            self.yc_clamp_max = float(yc_clamp_max_raw)
        self._reset_diag_episode()
        self.actor = DeterministicActor(state_dim, action_dim, max_action, ap_prior=ap_prior,
                                        num_devices=self.num_devices, num_rsu=self.num_rsu).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.q1 = Critic(state_dim, action_dim).to(self.device)
        self.q2 = Critic(state_dim, action_dim).to(self.device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.lr)
        if self.use_lagrangian:
            self.qc1 = CostCritic(state_dim, action_dim).to(self.device)
            self.qc2 = CostCritic(state_dim, action_dim).to(self.device)
            self.qc1_target = copy.deepcopy(self.qc1)
            self.qc2_target = copy.deepcopy(self.qc2)
            self.qc1_opt = torch.optim.Adam(self.qc1.parameters(), lr=cfg.lr)
            self.qc2_opt = torch.optim.Adam(self.qc2.parameters(), lr=cfg.lr)
        else:
            self.qc1 = None
            self.qc2 = None
            self.qc1_target = None
            self.qc2_target = None
            self.qc1_opt = None
            self.qc2_opt = None
        self.total_it = 0
        self.select_it = 0
        self._exploration_noise_scale = self._build_action_group_scale(
            rel_scale=self.cfg.exploration_noise_rel_scale,
            ap_scale=self.cfg.exploration_noise_ap_scale,
        )
        self._policy_noise_scale = torch.tensor(
            self._build_action_group_scale(
                rel_scale=self.cfg.policy_noise_rel_scale, ap_scale=self.cfg.policy_noise_ap_scale
            ),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

    def _project_action_for_critic(
        self, action: torch.Tensor, *, straight_through: bool
    ) -> torch.Tensor:
        return project_action_torch(
            action,
            num_devices=self.num_devices,
            num_rsu=self.num_rsu,
            mode=self.ap_projection_mode,
            straight_through=straight_through,
        )

    def _build_action_group_scale(self, rel_scale: float, ap_scale: float) -> np.ndarray:
        if self.ap_projection_mode in HYBRID_AP_MODES:
            expected = 2 * self.num_devices + self.num_devices * self.num_rsu
            if self.action_dim >= expected:
                return np.concatenate(
                    [
                        np.ones(self.num_devices, dtype=np.float32),
                        np.full(self.num_devices, rel_scale, dtype=np.float32),
                        np.full(self.num_devices * self.num_rsu, ap_scale, dtype=np.float32),
                        np.ones(max(self.action_dim - expected, 0), dtype=np.float32),
                    ]
                )
        if self.action_dim <= 0 or self.action_dim % 3 != 0:
            return np.ones(self.action_dim, dtype=np.float32)
        num_devices = self.action_dim // 3
        return np.concatenate(
            [
                np.ones(num_devices, dtype=np.float32),
                np.full(num_devices, rel_scale, dtype=np.float32),
                np.full(num_devices, ap_scale, dtype=np.float32),
            ]
        )

    def _noise_anneal_scale(self) -> float:
        final_scale = float(np.clip(self.cfg.noise_anneal_final_scale, 0.0, 1.0))
        anneal_steps = max(int(self.cfg.noise_anneal_steps), 1)
        progress = min(self.select_it / anneal_steps, 1.0)
        return 1.0 - (1.0 - final_scale) * progress

    def _reset_diag_episode(self):
        self._diag_episode = {
            "q1_pi_mean": [],
            "q2_pi_mean": [],
            "min_q_pi_mean": [],
            "q_disagreement_mean": [],
            "qc1_pi_mean": [],
            "qc2_pi_mean": [],
            "max_qc_pi_mean": [],
            "qc_disagreement_mean": [],
            "raw_qc_std": [],
            "raw_qc_p10": [],
            "raw_qc_p50": [],
            "raw_qc_p90": [],
            "reward_term_mean": [],
            "penalty_term_mean": [],
            "penalty_reward_ratio": [],
        }

    def _record_diag_actor_update(
        self,
        q1_pi: torch.Tensor,
        q2_pi: torch.Tensor,
        q_pi: torch.Tensor,
        qc1_pi: torch.Tensor,
        qc2_pi: torch.Tensor,
        qc_pi: torch.Tensor,
        reward_term: torch.Tensor,
        penalty_term: torch.Tensor,
    ):
        if not self.diag_enabled:
            return
        qc_flat = qc_pi.detach().flatten()
        qc_quantiles = torch.quantile(
            qc_flat, torch.tensor([0.1, 0.5, 0.9], dtype=qc_flat.dtype, device=qc_flat.device)
        )
        reward_term_val = float(reward_term.detach().item())
        penalty_term_val = float(penalty_term.detach().item())
        diag = self._diag_episode
        diag["q1_pi_mean"].append(float(q1_pi.detach().mean().item()))
        diag["q2_pi_mean"].append(float(q2_pi.detach().mean().item()))
        diag["min_q_pi_mean"].append(float(q_pi.detach().mean().item()))
        diag["q_disagreement_mean"].append(
            float((q1_pi.detach() - q2_pi.detach()).abs().mean().item())
        )
        diag["qc1_pi_mean"].append(float(qc1_pi.detach().mean().item()))
        diag["qc2_pi_mean"].append(float(qc2_pi.detach().mean().item()))
        diag["max_qc_pi_mean"].append(float(qc_pi.detach().mean().item()))
        diag["qc_disagreement_mean"].append(
            float((qc1_pi.detach() - qc2_pi.detach()).abs().mean().item())
        )
        diag["raw_qc_std"].append(float(qc_flat.std(unbiased=False).item()))
        diag["raw_qc_p10"].append(float(qc_quantiles[0].item()))
        diag["raw_qc_p50"].append(float(qc_quantiles[1].item()))
        diag["raw_qc_p90"].append(float(qc_quantiles[2].item()))
        diag["reward_term_mean"].append(reward_term_val)
        diag["penalty_term_mean"].append(penalty_term_val)
        diag["penalty_reward_ratio"].append(
            float(abs(penalty_term_val) / (abs(reward_term_val) + 1e-08))
        )

    def consume_diag_episode(
        self, episode: int, episode_cost: float, lambda_before: float, lambda_after: float
    ):
        if not self.diag_enabled:
            self._reset_diag_episode()
            return None
        diag = self._diag_episode
        actor_updates = len(diag["min_q_pi_mean"])

        def _mean(xs):
            return float(np.mean(xs)) if xs else None

        record = {
            "episode": int(episode),
            "actor_updates": int(actor_updates),
            "episode_cost": float(episode_cost),
            "cost_limit": float(self.cost_limit),
            "diff": float(episode_cost - self.cost_limit),
            "lambda_before": float(lambda_before),
            "lambda_after": float(lambda_after),
            "cost_coef": float(self.cost_coef),
            "q1_pi_mean": _mean(diag["q1_pi_mean"]),
            "q2_pi_mean": _mean(diag["q2_pi_mean"]),
            "min_q_pi_mean": _mean(diag["min_q_pi_mean"]),
            "q_disagreement_mean": _mean(diag["q_disagreement_mean"]),
            "qc1_pi_mean": _mean(diag["qc1_pi_mean"]),
            "qc2_pi_mean": _mean(diag["qc2_pi_mean"]),
            "max_qc_pi_mean": _mean(diag["max_qc_pi_mean"]),
            "qc_disagreement_mean": _mean(diag["qc_disagreement_mean"]),
            "raw_qc_std_mean": _mean(diag["raw_qc_std"]),
            "raw_qc_p10_mean": _mean(diag["raw_qc_p10"]),
            "raw_qc_p50_mean": _mean(diag["raw_qc_p50"]),
            "raw_qc_p90_mean": _mean(diag["raw_qc_p90"]),
            "reward_term_mean": _mean(diag["reward_term_mean"]),
            "penalty_term_mean": _mean(diag["penalty_term_mean"]),
            "penalty_reward_ratio_mean": _mean(diag["penalty_reward_ratio"]),
        }
        self._reset_diag_episode()
        return record

    @torch.no_grad()
    def select_action(self, state, deterministic: bool = False) -> np.ndarray:
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(s).squeeze(0).cpu().numpy()
        if not deterministic:
            self.select_it += 1
            noise_scale = self._noise_anneal_scale()
            noise = np.random.normal(0.0, self.cfg.exploration_noise, size=a.shape).astype(
                np.float32
            )
            noise = noise * self._exploration_noise_scale * noise_scale
            a = a + noise
        a = np.clip(a, -self.max_action, self.max_action)
        return a.astype(np.float32)

    def update(self, replay_buffer, batch_size: int = 256) -> dict:
        self.total_it += 1
        states, actions, rewards, costs, next_states, dones = replay_buffer.sample(batch_size)
        s = torch.tensor(states, dtype=torch.float32, device=self.device)
        a = torch.tensor(actions, dtype=torch.float32, device=self.device)
        r = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        c = torch.tensor(costs, dtype=torch.float32, device=self.device).unsqueeze(-1)
        s2 = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        d = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)
        with torch.no_grad():
            noise_scale = self._noise_anneal_scale()
            noise = (
                torch.randn_like(a)
                * (self.cfg.policy_noise * noise_scale)
                * self._policy_noise_scale
            ).clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
            a2 = (self.actor_target(s2) + noise).clamp(-self.max_action, self.max_action)
            a2 = self._project_action_for_critic(a2, straight_through=False)
            q1_t = self.q1_target(s2, a2)
            q2_t = self.q2_target(s2, a2)
            q_t = torch.min(q1_t, q2_t)
            y = r + (1.0 - d) * self.cfg.gamma * q_t
            if self.use_lagrangian:
                qc1_t = self.qc1_target(s2, a2)
                qc2_t = self.qc2_target(s2, a2)
                cost_now = torch.clamp(c, 0.0, 1.0)
                if self.cost_critic_target_mode == "discounted_return":
                    qc_t = torch.min(qc1_t, qc2_t)
                    y_c = cost_now + (1.0 - d) * self.cfg.gamma * qc_t
                elif self.cost_critic_target_mode == "discounted_average":
                    qc_t = torch.clamp(torch.max(qc1_t, qc2_t), min=0.0)
                    y_c = (1.0 - self.cfg.gamma) * cost_now + self.cfg.gamma * (1.0 - d) * qc_t
                else:
                    y_c = cost_now
                if self.yc_clamp_max is None:
                    y_c = torch.clamp(y_c, min=0.0)
                else:
                    y_c = torch.clamp(y_c, 0.0, self.yc_clamp_max)
        q1 = self.q1(s, a)
        q2 = self.q2(s, a)
        loss_q1 = F.mse_loss(q1, y)
        loss_q2 = F.mse_loss(q2, y)
        self.q1_opt.zero_grad()
        loss_q1.backward()
        self.q1_opt.step()
        self.q2_opt.zero_grad()
        loss_q2.backward()
        self.q2_opt.step()
        if self.use_lagrangian:
            qc1 = self.qc1(s, a)
            qc2 = self.qc2(s, a)
            loss_qc1 = F.mse_loss(qc1, y_c)
            loss_qc2 = F.mse_loss(qc2, y_c)
            loss_qc = loss_qc1 + loss_qc2
            self.qc1_opt.zero_grad()
            self.qc2_opt.zero_grad()
            loss_qc.backward()
            self.qc1_opt.step()
            self.qc2_opt.step()
        else:
            loss_qc = torch.tensor(0.0, device=self.device)
        actor_loss = torch.tensor(0.0, device=self.device)
        constraint_diag_valid = 0.0
        constraint_active_update = 0.0
        constraint_active_frac = 0.0
        constraint_violation_mean = 0.0
        constraint_violation_max = 0.0
        cost_pi_mean = 0.0
        cost_pi_scaled_mean = 0.0
        lambda_update_count = 0.0
        lambda_positive_update_count = 0.0
        if self.total_it % self.cfg.policy_delay == 0:
            pi = self.actor(s)
            pi = self._project_action_for_critic(pi, straight_through=True)
            q1_pi = self.q1(s, pi)
            q2_pi = self.q2(s, pi)
            q_pi = torch.min(q1_pi, q2_pi)
            if self.use_lagrangian:
                qc1_pi = self.qc1(s, pi)
                qc2_pi = self.qc2(s, pi)
                qc_pi = torch.max(qc1_pi, qc2_pi)
                constraint_violation = torch.clamp(qc_pi - self.cost_limit, min=0.0)
                constraint_violation_det = constraint_violation.detach()
                cost_pi_mean = float(qc_pi.detach().mean().item())
                cost_pi_scaled_mean = float(qc_pi.detach().mean().item())
                constraint_active_frac = float((constraint_violation_det > 0).float().mean().item())
                constraint_violation_mean = float(constraint_violation_det.mean().item())
                constraint_violation_max = float(constraint_violation_det.max().item())
                constraint_active_update = 1.0 if constraint_violation_max > 0.0 else 0.0
                constraint_diag_valid = 1.0
                reward_term = -q_pi.mean()
                penalty_term = self.lambda_val * self.cost_coef * constraint_violation.mean()
                self._record_diag_actor_update(
                    q1_pi=q1_pi,
                    q2_pi=q2_pi,
                    q_pi=q_pi,
                    qc1_pi=qc1_pi,
                    qc2_pi=qc2_pi,
                    qc_pi=qc_pi,
                    reward_term=reward_term,
                    penalty_term=penalty_term,
                )
                actor_loss = reward_term + penalty_term
            else:
                actor_loss = -q_pi.mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.q1, self.q1_target)
            self._soft_update(self.q2, self.q2_target)
            if self.use_lagrangian:
                self._soft_update(self.qc1, self.qc1_target)
                self._soft_update(self.qc2, self.qc2_target)
        if self.use_lagrangian and self.update_lambda_in_batch:
            mean_c_batch = float(c.mean().item())
            diff = mean_c_batch - self.cost_limit
            lambda_update_count = 1.0
            lambda_positive_update_count = 1.0 if diff > 0.0 else 0.0
            if self.lambda_mode == "fixed":
                new_lambda = self.fixed_lambda
            else:
                if self.lambda_mode == "slow":
                    effective_lr = self.lambda_lr * self.slow_lambda_factor
                else:
                    effective_lr = self.lambda_lr * (1.5 if diff > 0 else 0.8)
                new_lambda = self.lambda_val + effective_lr * diff
            new_lambda = max(self.min_lambda, min(self.max_lambda, new_lambda))
            self.lambda_val = float(new_lambda)
        return {
            "actor_loss": float(actor_loss.item()),
            "loss_q1": float(loss_q1.item()),
            "loss_q2": float(loss_q2.item()),
            "loss_qc": float(loss_qc.item()),
            "lambda": self.lambda_val,
            "alpha_loss": 0.0,
            "constraint_diag_valid": float(constraint_diag_valid),
            "constraint_active_update": float(constraint_active_update),
            "constraint_active_frac": float(constraint_active_frac),
            "constraint_violation_mean": float(constraint_violation_mean),
            "constraint_violation_max": float(constraint_violation_max),
            "cost_pi_mean": float(cost_pi_mean),
            "cost_pi_scaled_mean": float(cost_pi_scaled_mean),
            "lambda_update_count": float(lambda_update_count),
            "lambda_positive_update_count": float(lambda_positive_update_count),
        }

    def update_lambda(self, episode_cost: float) -> float:
        if not self.use_lagrangian:
            return float(self.lambda_val)
        if self.lambda_mode == "fixed":
            new_lambda = self.fixed_lambda
        else:
            diff = float(episode_cost - self.cost_limit)
            if self.lambda_mode == "slow":
                effective_lr = self.lambda_lr * self.slow_lambda_factor
            else:
                effective_lr = self.lambda_lr * (1.5 if diff > 0 else 0.8)
            new_lambda = self.lambda_val + effective_lr * diff
        new_lambda = max(self.min_lambda, min(self.max_lambda, new_lambda))
        self.lambda_val = float(new_lambda)
        return self.lambda_val

    def _soft_update(self, net: nn.Module, target_net: nn.Module):
        tau = self.cfg.tau
        for p, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
