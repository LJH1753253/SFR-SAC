import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from agents.action_projection import project_action_torch
from agents.networks import Critic, CostCritic, HybridActor
from agents.ap_policy_controls import FrozenAPPolicy, sample_actor_entropy, validate_entropy_mode


class SafeSACAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action=1.0,
        action_limit=None,
        gamma=0.99,
        tau=0.005,
        alpha=0.02,
        lr=0.003,
        cost_limit=0.09,
        lambda_lr=0.0002,
        device="cpu",
        use_lagrangian=True,
        auto_alpha=False,
        target_entropy=None,
        alpha_lr=0.0005,
        cost_coef=3.5,
        update_lambda_in_batch=True,
        lambda_mode="adaptive",
        fixed_lambda=1.0,
        initial_lambda=1.0,
        slow_lambda_factor=0.2,
        num_devices=None,
        num_rsu=1,
        ap_projection_mode="legacy",
        ap_prior=None,
        cost_critic_target_mode="immediate",
        action_mode="legacy",
        gumbel_tau=1.0,
        gumbel_tau_min=0.3,
        ap_entropy_weight=1.0,
        penalty_mode="hinge",
        cost_critic_factory=None,
        cost_delivery_loss_weight=1.0,
        reward_critic_factory=None,
        energy_component_loss_weight=1.0,
        dual_cost_source="replay_cost",
        dual_risk_mode="mean",
        excess_budget=0.0002,
        actor_factory=None,
        actor_grad_clip=None,
        actor_update_interval=1,
        freeze_alpha=False,
        ap_entropy_mode="legacy_sampled",
        freeze_ap_policy=False,
        matched_entropy_temperature=False,
        ap_gradient_mode="straight_through",
        ap_value_devices=4,
        ap_value_chunk=512,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.use_lagrangian = use_lagrangian
        self.structured_energy_enabled = reward_critic_factory is not None
        self.energy_component_loss_weight = float(energy_component_loss_weight)
        if self.structured_energy_enabled and gamma != 0.0:
            raise ValueError("Physical energy critic is an immediate predictor; gamma must be zero")
        # "hinge": lambda*beta_c*[Qc-d]^+ (manuscript Eq. 72, default, archived behaviour)
        # "linear": lambda*beta_c*Qc (untruncated Lagrangian penalty, ablation for R1.2)
        self.penalty_mode = str(penalty_mode).lower()
        if self.penalty_mode not in {"hinge", "linear"}:
            raise ValueError("penalty_mode must be 'hinge' or 'linear'")
        self.dual_cost_source = str(dual_cost_source)
        self.dual_risk_mode = str(dual_risk_mode)
        self.excess_budget = float(excess_budget)
        if self.dual_cost_source not in {"replay_cost", "policy_prediction"}:
            raise ValueError("Unknown dual_cost_source")
        if self.dual_risk_mode not in {"mean", "excess"} or self.excess_budget < 0:
            raise ValueError("Invalid dual risk mode or excess budget")
        if self.dual_risk_mode == "excess" and self.penalty_mode != "hinge":
            raise ValueError("Excess-risk dual requires a hinge actor penalty")
        if not update_lambda_in_batch and (self.dual_cost_source != "replay_cost" or self.dual_risk_mode != "mean"):
            raise ValueError("Alternative dual statistics are supported for batch updates only")
        self.action_dim = float(action_dim)
        self.update_step = 0
        if actor_grad_clip is not None and (not np.isfinite(actor_grad_clip) or actor_grad_clip <= 0):
            raise ValueError("actor_grad_clip must be positive or None")
        if int(actor_update_interval) != actor_update_interval or actor_update_interval < 1:
            raise ValueError("actor_update_interval must be a positive integer")
        self.actor_grad_clip = actor_grad_clip
        self.actor_update_interval = int(actor_update_interval)
        self.freeze_alpha = bool(freeze_alpha)
        validate_entropy_mode(ap_entropy_mode, gamma)
        self.ap_entropy_mode = ap_entropy_mode
        self.freeze_ap_policy = bool(freeze_ap_policy)
        self.matched_entropy_temperature = bool(matched_entropy_temperature)
        self.ap_gradient_mode = ap_gradient_mode
        self.ap_value_devices, self.ap_value_chunk = int(ap_value_devices), int(ap_value_chunk)
        if ap_gradient_mode not in {"straight_through", "discrete_values"}:
            raise ValueError("Unknown AP gradient mode")
        if min(self.ap_value_devices, self.ap_value_chunk) <= 0:
            raise ValueError("AP candidate sizes must be positive")
        if ap_gradient_mode == "discrete_values" and (gamma != 0 or self.freeze_ap_policy):
            raise ValueError("Discrete AP values require gamma=0 and a trainable AP policy")
        self.auto_alpha = auto_alpha
        if self.auto_alpha:
            self.log_alpha = torch.tensor(
                np.log(alpha), dtype=torch.float32, device=self.device, requires_grad=True
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
            if target_entropy is None:
                self.target_entropy = -float(action_dim) / 2.0
            else:
                self.target_entropy = float(target_entropy)
            self.alpha = float(alpha)
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
            self.target_entropy = None
            self.alpha = float(alpha)
        if action_limit is not None:
            max_action = action_limit
        self.max_action = max_action
        self.num_devices = int(num_devices) if num_devices is not None else int(action_dim // 3)
        self.num_rsu = int(num_rsu)
        self.action_mode = str(action_mode).lower()
        self.hybrid_ap_enabled = self.action_mode in {"hybrid_ap_gumbel", "hybrid_ap_onehot"}
        self.ap_projection_mode = str(ap_projection_mode).lower()
        self.cost_critic_target_mode = str(cost_critic_target_mode).lower()
        self.structured_cost_enabled = cost_critic_factory is not None
        self.cost_delivery_loss_weight = float(cost_delivery_loss_weight)
        if self.structured_cost_enabled and self.cost_critic_target_mode != "immediate":
            raise ValueError("Service-flow critic supports immediate targets only")
        self.gumbel_tau = float(gumbel_tau)
        self.gumbel_tau_min = float(gumbel_tau_min)
        self.ap_entropy_weight = float(ap_entropy_weight)
        self.entropy_action_dim = float(
            2 * self.num_devices if self.hybrid_ap_enabled else action_dim
        )
        valid_cost_target_modes = {"immediate", "discounted_average"}
        if self.cost_critic_target_mode not in valid_cost_target_modes:
            raise ValueError(
                f"Unknown cost_critic_target_mode={self.cost_critic_target_mode}, expected one of {valid_cost_target_modes}"
            )
        self.actor = HybridActor(
            state_dim,
            self.num_devices,
            self.num_rsu,
            max_action=self.max_action,
            gumbel_tau=self.gumbel_tau,
            gumbel_tau_min=self.gumbel_tau_min,
            ap_entropy_weight=self.ap_entropy_weight,
            ap_prior=ap_prior,
        ).to(device)
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)
        cost_factory = CostCritic if cost_critic_factory is None else cost_critic_factory
        self.cost_critic_1 = cost_factory(state_dim, action_dim).to(device)
        self.cost_critic_2 = cost_factory(state_dim, action_dim).to(device)
        self.cost_critic_1_target = cost_factory(state_dim, action_dim).to(device)
        self.cost_critic_2_target = cost_factory(state_dim, action_dim).to(device)
        if self.structured_energy_enabled:
            with torch.random.fork_rng(devices=[]):
                self.critic_1 = reward_critic_factory(state_dim, action_dim).to(device)
                self.critic_2 = reward_critic_factory(state_dim, action_dim).to(device)
                self.critic_1_target = reward_critic_factory(state_dim, action_dim).to(device)
                self.critic_2_target = reward_critic_factory(state_dim, action_dim).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.cost_critic_1_target.load_state_dict(self.cost_critic_1.state_dict())
        self.cost_critic_2_target.load_state_dict(self.cost_critic_2.state_dict())
        if actor_factory is not None:
            with torch.random.fork_rng(devices=[]):
                self.actor = actor_factory(
                    state_dim, self.num_devices, self.num_rsu, max_action=self.max_action,
                    gumbel_tau=self.gumbel_tau, gumbel_tau_min=self.gumbel_tau_min,
                    ap_entropy_weight=self.ap_entropy_weight, ap_prior=ap_prior,
                ).to(device)
        if self.freeze_ap_policy:
            self.actor = FrozenAPPolicy(self.actor)
        self.actor_optimizer = optim.Adam([p for p in self.actor.parameters() if p.requires_grad], lr=lr, weight_decay=1e-05)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr, weight_decay=1e-05)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr, weight_decay=1e-05)
        self.cost_critic_1_optimizer = optim.Adam(
            self.cost_critic_1.parameters(), lr=lr, weight_decay=1e-05
        )
        self.cost_critic_2_optimizer = optim.Adam(
            self.cost_critic_2.parameters(), lr=lr, weight_decay=1e-05
        )
        self.cost_limit = float(cost_limit)
        self.lambda_lr = float(lambda_lr)
        self.lambda_mode = str(lambda_mode).lower()
        self.fixed_lambda = float(fixed_lambda)
        self.initial_lambda = float(initial_lambda)
        self.slow_lambda_factor = float(slow_lambda_factor)
        self.cost_coef = float(cost_coef)
        self.max_lambda = 50.0
        self.min_lambda = 0.01
        init_lambda = self.fixed_lambda if self.lambda_mode == "fixed" else self.initial_lambda
        init_lambda = max(self.min_lambda, min(self.max_lambda, init_lambda))
        self.lam = torch.tensor(
            [init_lambda], dtype=torch.float32, requires_grad=False, device=device
        )
        self.lambda_val = float(init_lambda)
        self.update_lambda_in_batch = bool(update_lambda_in_batch)
        valid_lambda_modes = {"adaptive", "fixed", "slow"}
        if self.lambda_mode not in valid_lambda_modes:
            raise ValueError(
                f"Unknown lambda_mode={self.lambda_mode}, expected one of {valid_lambda_modes}"
            )

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

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _ = self.actor.sample(state, deterministic=deterministic)
        return action.detach().cpu().numpy()[0]

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def dual_statistic(self, replay_cost, policy_cost):
        values = replay_cost if self.dual_cost_source == "replay_cost" else policy_cost.detach()
        if self.dual_risk_mode == "excess":
            return torch.clamp(values - self.cost_limit, min=0.0).mean(), self.excess_budget
        return values.mean(), self.cost_limit

    def update(self, replay_buffer, batch_size=256):
        delivery_labels = None
        energy_labels = None
        if self.structured_energy_enabled:
            transition, delivery_labels, energy_labels = replay_buffer.sample_with_physics(batch_size)
            s, a, r, c, s2, d = transition
            if delivery_labels is not None:
                delivery_labels = torch.as_tensor(delivery_labels, dtype=torch.float32, device=self.device)
            energy_labels = torch.as_tensor(energy_labels, dtype=torch.float32, device=self.device)
        elif self.structured_cost_enabled:
            transition, delivery_labels = replay_buffer.sample_with_delivery(batch_size)
            s, a, r, c, s2, d = transition
            delivery_labels = torch.as_tensor(delivery_labels, dtype=torch.float32, device=self.device)
        else:
            s, a, r, c, s2, d = replay_buffer.sample(batch_size)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device).unsqueeze(1)
        s2 = torch.FloatTensor(s2).to(self.device)
        c = torch.FloatTensor(c).to(self.device).unsqueeze(1)
        d = torch.FloatTensor(d).to(self.device).unsqueeze(1)
        with torch.no_grad():
            a2, logp_a2 = self.actor.sample(s2)
            a2 = self._project_action_for_critic(a2, straight_through=False)
            q1_next = self.critic_1_target(s2, a2)
            q2_next = self.critic_2_target(s2, a2)
            entropy_term_next = self.alpha * (logp_a2 / self.entropy_action_dim)
            q_next = torch.min(q1_next, q2_next) - entropy_term_next
            q_target = r + self.gamma * (1.0 - d) * q_next
            qc1_next = self.cost_critic_1_target(s2, a2)
            qc2_next = self.cost_critic_2_target(s2, a2)
            cost_now = torch.clamp(c, 0.0, 1.0)
            if self.cost_critic_target_mode == "discounted_average":
                qc_next = torch.clamp(torch.max(qc1_next, qc2_next), min=0.0)
                cost_target = (1.0 - self.gamma) * cost_now + self.gamma * (1.0 - d) * qc_next
            else:
                cost_target = cost_now
        energy_loss = torch.zeros((), device=self.device)
        if self.structured_energy_enabled:
            q1, components1 = self.critic_1.predict_energy(s, a)
            q2, components2 = self.critic_2.predict_energy(s, a)
            energy_loss1 = F.smooth_l1_loss(components1, energy_labels, beta=0.1)
            energy_loss2 = F.smooth_l1_loss(components2, energy_labels, beta=0.1)
            energy_loss = 0.5 * (energy_loss1 + energy_loss2)
        else:
            q1 = self.critic_1(s, a)
            q2 = self.critic_2(s, a)
        critic_1_loss = F.smooth_l1_loss(q1, q_target, beta=1.0)
        critic_2_loss = F.smooth_l1_loss(q2, q_target, beta=1.0)
        if self.structured_energy_enabled:
            critic_1_loss = critic_1_loss + self.energy_component_loss_weight * energy_loss1
            critic_2_loss = critic_2_loss + self.energy_component_loss_weight * energy_loss2
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=1.0)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=1.0)
        self.critic_2_optimizer.step()
        delivery_loss = torch.zeros((), device=self.device)
        if self.structured_cost_enabled:
            qc1, delivery1 = self.cost_critic_1.predict(s, a)
            qc2, delivery2 = self.cost_critic_2.predict(s, a)
            loss_delivery1 = F.smooth_l1_loss(delivery1, delivery_labels, beta=0.1)
            loss_delivery2 = F.smooth_l1_loss(delivery2, delivery_labels, beta=0.1)
            delivery_loss = 0.5 * (loss_delivery1 + loss_delivery2)
        else:
            qc1 = self.cost_critic_1(s, a)
            qc2 = self.cost_critic_2(s, a)
        loss_c1 = F.smooth_l1_loss(qc1, cost_target, beta=0.1)
        loss_c2 = F.smooth_l1_loss(qc2, cost_target, beta=0.1)
        if self.structured_cost_enabled:
            loss_c1 = loss_c1 + self.cost_delivery_loss_weight * loss_delivery1
            loss_c2 = loss_c2 + self.cost_delivery_loss_weight * loss_delivery2
        cost_critic_loss = 0.5 * (loss_c1 + loss_c2)
        self.cost_critic_1_optimizer.zero_grad()
        loss_c1.backward()
        torch.nn.utils.clip_grad_norm_(self.cost_critic_1.parameters(), max_norm=1.0)
        self.cost_critic_1_optimizer.step()
        self.cost_critic_2_optimizer.zero_grad()
        loss_c2.backward()
        torch.nn.utils.clip_grad_norm_(self.cost_critic_2.parameters(), max_norm=1.0)
        self.cost_critic_2_optimizer.step()
        a_pi, logp_pi, actor_entropy = sample_actor_entropy(
            self.actor, s, self.ap_entropy_mode, self.entropy_action_dim)
        a_pi = self._project_action_for_critic(a_pi, straight_through=True)
        if self.ap_gradient_mode == "discrete_values":
            split = 2*self.num_devices
            a_pi = torch.cat((a_pi[:, :split], a_pi[:, split:].detach()), 1)
        q1_pi = self.critic_1(s, a_pi)
        q2_pi = self.critic_2(s, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        cost_pi_1 = self.cost_critic_1(s, a_pi)
        cost_pi_2 = self.cost_critic_2(s, a_pi)
        cost_pi = torch.max(cost_pi_1, cost_pi_2)
        cost_pi_scaled = torch.clamp(cost_pi, min=0.0)
        cost_pi_mean = cost_pi.detach().mean().item()
        entropy_term = self.alpha * actor_entropy
        constraint_active_frac = 0.0
        constraint_violation_mean = 0.0
        constraint_violation_max = 0.0
        constraint_active_update = 0.0
        cost_pi_scaled_mean = 0.0
        lambda_update_count = 0.0
        lambda_positive_update_count = 0.0
        dual_statistic = 0.0
        dual_budget = 0.0
        if self.use_lagrangian:
            lambda_detach = self.lam.detach()
            if self.penalty_mode == "linear":
                constraint_violation = cost_pi_scaled - self.cost_limit
            else:
                constraint_violation = torch.clamp(cost_pi_scaled - self.cost_limit, min=0.0)
            constraint_violation_det = constraint_violation.detach()
            cost_pi_scaled_mean = float(cost_pi_scaled.detach().mean().item())
            constraint_active_frac = float((constraint_violation_det > 0).float().mean().item())
            constraint_violation_mean = float(constraint_violation_det.mean().item())
            constraint_violation_max = float(constraint_violation_det.max().item())
            constraint_active_update = 1.0 if constraint_violation_max > 0.0 else 0.0
            safe_term = lambda_detach * self.cost_coef * constraint_violation
            actor_loss = (entropy_term - q_pi + safe_term).mean()
        else:
            actor_loss = (entropy_term - q_pi).mean()
        ap_value_loss = torch.zeros((), device=s.device)
        if self.ap_gradient_mode == "discrete_values":
            from agents.cloud_actor import categorical_value_loss
            ap_value_loss = categorical_value_loss(self, s, a_pi)
            actor_loss = actor_loss + ap_value_loss
        actor_updated = (self.update_step + 1) % self.actor_update_interval == 0
        actor_grad_norm = 0.0
        actor_clipped = False
        self.actor_optimizer.zero_grad()
        if actor_updated:
            actor_loss.backward()
            actor_grad_norm = float(torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), float("inf") if self.actor_grad_clip is None else self.actor_grad_clip
            ).item())
            actor_clipped = self.actor_grad_clip is not None and actor_grad_norm > self.actor_grad_clip
            self.actor_optimizer.step()
        alpha_loss_value = 0.0
        driver = (actor_entropy + self.target_entropy/self.entropy_action_dim
                  if self.auto_alpha and self.matched_entropy_temperature else
                  logp_pi + self.target_entropy if self.auto_alpha else logp_pi.detach()*0.)
        if self.auto_alpha and not self.freeze_alpha:
            alpha_loss = -(self.log_alpha * driver.detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = float(self.log_alpha.exp().item())
            alpha_loss_value = float(alpha_loss.item())
        if self.use_lagrangian and self.update_lambda_in_batch:
            with torch.no_grad():
                statistic, dual_budget = self.dual_statistic(c, cost_pi_scaled)
                dual_statistic = float(statistic.item())
                diff = dual_statistic - dual_budget
                lambda_update_count = 1.0
                lambda_positive_update_count = 1.0 if diff > 0.0 else 0.0
                if self.lambda_mode == "fixed":
                    new_lambda = self.fixed_lambda
                elif self.lambda_mode == "slow":
                    adaptive_lr = self.lambda_lr * self.slow_lambda_factor
                    new_lambda = self.lam.item() + adaptive_lr * diff
                else:
                    adaptive_lr = self.lambda_lr * (1.5 if diff > 0 else 0.8)
                    new_lambda = self.lam.item() + adaptive_lr * diff
                new_lambda = max(self.min_lambda, min(self.max_lambda, new_lambda))
                self.lam[...] = new_lambda
                self.lambda_val = float(new_lambda)
        else:
            self.lambda_val = float(self.lam.item())
        for param, target_param in zip(
            self.critic_1.parameters(), self.critic_1_target.parameters()
        ):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for param, target_param in zip(
            self.critic_2.parameters(), self.critic_2_target.parameters()
        ):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for param, target_param in zip(
            self.cost_critic_1.parameters(), self.cost_critic_1_target.parameters()
        ):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for param, target_param in zip(
            self.cost_critic_2.parameters(), self.cost_critic_2_target.parameters()
        ):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        self.update_step += 1
        actor_sample_info = getattr(self.actor, "last_sample_info", {})
        return {
            "actor_loss": float(actor_loss.item()),
            "ap_value_loss": float(ap_value_loss.detach().item()),
            "actor_updated": float(actor_updated),
            "actor_grad_norm": actor_grad_norm,
            "actor_clipped": float(actor_clipped),
            "actor_entropy_term": float(entropy_term.detach().mean().item()),
            "actor_energy_term": float(-q_pi.detach().mean().item()),
            "actor_constraint_term": float(safe_term.detach().mean().item()) if self.use_lagrangian else 0.0,
            "cost_critic_disagreement": float((cost_pi_1.detach() - cost_pi_2.detach()).abs().mean().item()),
            "entropy_driver": float(driver.detach().mean().item()) if self.auto_alpha else 0.0,
            "continuous_log_prob_per_dim": float(actor_sample_info.get("log_prob_cont", logp_pi.detach()).mean().item()) / self.entropy_action_dim,
            "categorical_log_prob_per_device": float(actor_sample_info.get("log_prob_ap_norm", torch.tensor(0.)).mean().item()),
            "replay_age_steps": float(getattr(replay_buffer, "last_sample_stats", {}).get("mean_age_steps", 0.0)),
            "replay_recent_fraction": float(getattr(replay_buffer, "last_sample_stats", {}).get("recent_fraction", 0.0)),
            "loss_q1": float(critic_1_loss.item()),
            "loss_q2": float(critic_2_loss.item()),
            "loss_qc": float(cost_critic_loss.item()),
            "loss_delivery": float(delivery_loss.item()),
            "loss_energy_components": float(energy_loss.item()),
            "alpha_loss": float(alpha_loss_value),
            "constraint_diag_valid": 1.0 if self.use_lagrangian else 0.0,
            "constraint_active_update": float(constraint_active_update),
            "constraint_active_frac": float(constraint_active_frac),
            "constraint_violation_mean": float(constraint_violation_mean),
            "constraint_violation_max": float(constraint_violation_max),
            "cost_pi_mean": float(cost_pi_mean),
            "cost_pi_scaled_mean": float(cost_pi_scaled_mean),
            "lambda_update_count": float(lambda_update_count),
            "lambda_positive_update_count": float(lambda_positive_update_count),
            "dual_statistic": float(dual_statistic),
            "dual_budget": float(dual_budget),
            "dual_replay_cost_mean": float(c.detach().mean().item()),
            "ap_entropy": float(
                actor_sample_info.get("ap_entropy", torch.tensor(0.0)).mean().item()
            )
            if self.hybrid_ap_enabled
            else 0.0,
            "ap_max_prob": float(
                actor_sample_info.get("ap_max_prob", torch.tensor(0.0)).mean().item()
            )
            if self.hybrid_ap_enabled
            else 1.0,
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
        self.lam[...] = new_lambda
        self.lambda_val = float(new_lambda)
        return self.lambda_val


class VanillaSACAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action=1.0,
        action_limit=None,
        gamma=0.99,
        tau=0.01,
        alpha=0.02,
        lr=5e-05,
        device="cpu",
        auto_alpha=True,
        target_entropy=None,
        alpha_lr=0.0001,
        num_devices=None,
        num_rsu=1,
        ap_projection_mode="legacy",
        ap_prior=None,
        action_mode="legacy",
        gumbel_tau=1.0,
        gumbel_tau_min=0.3,
        ap_entropy_weight=1.0,
        actor_factory=None,
        ap_entropy_mode="legacy_sampled",
        freeze_ap_policy=False,
        matched_entropy_temperature=False,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        self.auto_alpha = auto_alpha
        validate_entropy_mode(ap_entropy_mode, gamma)
        self.ap_entropy_mode = ap_entropy_mode
        self.freeze_ap_policy = bool(freeze_ap_policy)
        self.matched_entropy_temperature = bool(matched_entropy_temperature)
        if self.auto_alpha:
            self.log_alpha = torch.tensor(
                np.log(alpha), dtype=torch.float32, device=self.device, requires_grad=True
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
            if target_entropy is None:
                self.target_entropy = -float(action_dim) / 2
            else:
                self.target_entropy = float(target_entropy)
            self.alpha = float(alpha)
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
            self.target_entropy = None
            self.alpha = float(alpha)
        if action_limit is not None:
            max_action = action_limit
        self.max_action = max_action
        self.num_devices = int(num_devices) if num_devices is not None else int(action_dim // 3)
        self.num_rsu = int(num_rsu)
        self.action_mode = str(action_mode).lower()
        self.hybrid_ap_enabled = self.action_mode in {"hybrid_ap_gumbel", "hybrid_ap_onehot"}
        self.ap_projection_mode = str(ap_projection_mode).lower()
        self.gumbel_tau = float(gumbel_tau)
        self.gumbel_tau_min = float(gumbel_tau_min)
        self.ap_entropy_weight = float(ap_entropy_weight)
        self.entropy_action_dim = float(
            2 * self.num_devices if self.hybrid_ap_enabled else action_dim
        )
        self.actor = HybridActor(
            state_dim,
            self.num_devices,
            self.num_rsu,
            max_action=self.max_action,
            gumbel_tau=self.gumbel_tau,
            gumbel_tau_min=self.gumbel_tau_min,
            ap_entropy_weight=self.ap_entropy_weight,
            ap_prior=ap_prior,
        ).to(device)
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        if actor_factory is not None:
            with torch.random.fork_rng(devices=[]):
                self.actor = actor_factory(
                    state_dim, self.num_devices, self.num_rsu, max_action=self.max_action,
                    gumbel_tau=self.gumbel_tau, gumbel_tau_min=self.gumbel_tau_min,
                    ap_entropy_weight=self.ap_entropy_weight, ap_prior=ap_prior,
                ).to(device)
        if self.freeze_ap_policy:
            self.actor = FrozenAPPolicy(self.actor)
        self.actor_optimizer = optim.Adam([p for p in self.actor.parameters() if p.requires_grad], lr=lr, weight_decay=1e-05)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr, weight_decay=1e-05)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr, weight_decay=1e-05)
        self.lambda_val = 0.0

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

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor.sample(state, deterministic=deterministic)
        return action.cpu().numpy()[0]

    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update(self, replay_buffer, batch_size=256):
        s, a, r, c, s2, d = replay_buffer.sample(batch_size)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device).unsqueeze(1)
        s2 = torch.FloatTensor(s2).to(self.device)
        d = torch.FloatTensor(d).to(self.device).unsqueeze(1)
        with torch.no_grad():
            a2, logp_a2 = self.actor.sample(s2)
            a2 = self._project_action_for_critic(a2, straight_through=False)
            q1_next = self.critic_1_target(s2, a2)
            q2_next = self.critic_2_target(s2, a2)
            entropy_term = self.alpha * (logp_a2 / self.entropy_action_dim)
            q_next = torch.min(q1_next, q2_next) - entropy_term
            q_target = r + self.gamma * (1 - d) * q_next
        q1 = self.critic_1(s, a)
        q2 = self.critic_2(s, a)
        critic_1_loss = F.smooth_l1_loss(q1, q_target, beta=1.0)
        critic_2_loss = F.smooth_l1_loss(q2, q_target, beta=1.0)
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=1.0)
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=1.0)
        self.critic_2_optimizer.step()
        a_pi, logp_pi, actor_entropy = sample_actor_entropy(
            self.actor, s, self.ap_entropy_mode, self.entropy_action_dim)
        a_pi = self._project_action_for_critic(a_pi, straight_through=True)
        q1_pi = self.critic_1(s, a_pi)
        q2_pi = self.critic_2(s, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * actor_entropy - q_pi).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        alpha_loss_value = 0.0
        if self.auto_alpha:
            driver = (actor_entropy + self.target_entropy/self.entropy_action_dim
                      if self.matched_entropy_temperature else logp_pi + self.target_entropy)
            alpha_loss = -(self.log_alpha * driver.detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = float(self.log_alpha.exp().item())
            alpha_loss_value = float(alpha_loss.item())
        self.soft_update(self.critic_1, self.critic_1_target)
        self.soft_update(self.critic_2, self.critic_2_target)
        actor_sample_info = getattr(self.actor, "last_sample_info", {})
        return {
            "actor_loss": actor_loss.item(),
            "loss_q1": critic_1_loss.item(),
            "loss_q2": critic_2_loss.item(),
            "loss_qc": 0.0,
            "lambda": self.lambda_val,
            "alpha_loss": alpha_loss_value,
            "ap_entropy": float(
                actor_sample_info.get("ap_entropy", torch.tensor(0.0)).mean().item()
            )
            if self.hybrid_ap_enabled
            else 0.0,
            "ap_max_prob": float(
                actor_sample_info.get("ap_max_prob", torch.tensor(0.0)).mean().item()
            )
            if self.hybrid_ap_enabled
            else 1.0,
        }
