import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class APLinkPrior(nn.Module):
    """Channel-aware AP logits (v2): kappa * (link quality - best link quality) per ED.

    `start` is the offset of the N*M normalized link-SNR block in the state. The prior is zero for
    each ED's best link and negative for the others; kappa is learned (log-parameterized). Disabled
    unless an agent is built with `ap_prior=...`, so archived checkpoints load unchanged.
    """

    def __init__(self, start, num_devices, num_rsu, init_scale=25.0, learnable=True):
        super().__init__()
        self.start, self.N, self.M = int(start), int(num_devices), int(num_rsu)
        value = torch.tensor(float(np.log(init_scale)))
        if learnable:
            self.log_scale = nn.Parameter(value)
        else:
            self.register_buffer("log_scale", value)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        q = state[:, self.start : self.start + self.N * self.M].reshape(-1, self.N, self.M)
        return self.log_scale.exp() * (q - q.max(dim=-1, keepdim=True).values)


def make_ap_prior(ap_prior, num_devices, num_rsu):
    if not ap_prior:
        return None
    return APLinkPrior(ap_prior["start"], num_devices, num_rsu, ap_prior.get("init_scale", 25.0),
                       ap_prior.get("learnable", True))


class HybridActor(nn.Module):
    def __init__(
        self,
        state_dim,
        num_devices,
        num_rsu,
        max_action=1.0,
        gumbel_tau=1.0,
        gumbel_tau_min=0.3,
        ap_entropy_weight=1.0,
        ap_prior=None,
    ):
        super().__init__()
        self.num_devices = int(num_devices)
        self.num_rsu = int(num_rsu)
        if self.num_devices <= 0:
            raise ValueError("num_devices must be positive for HybridActor")
        if self.num_rsu <= 0:
            raise ValueError("num_rsu must be positive for HybridActor")
        self.continuous_dim = 2 * self.num_devices
        self.ap_dim = self.num_devices * self.num_rsu
        self.action_dim = self.continuous_dim + self.ap_dim
        self.max_action = float(max_action)
        self.gumbel_tau = float(gumbel_tau)
        self.gumbel_tau_min = float(gumbel_tau_min)
        self.ap_entropy_weight = float(ap_entropy_weight)
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.last_sample_info = {}
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.cont_mean_layer = nn.Linear(256, self.continuous_dim)
        self.cont_log_std_layer = nn.Linear(256, self.continuous_dim)
        self.ap_logits_layer = nn.Linear(256, self.ap_dim)
        self.ap_prior = make_ap_prior(ap_prior, self.num_devices, self.num_rsu)

    def forward(self, state: torch.Tensor):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.cont_mean_layer(x)
        log_std = self.cont_log_std_layer(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        ap_logits = self.ap_logits_layer(x).view(-1, self.num_devices, self.num_rsu)
        if self.ap_prior is not None:
            ap_logits = ap_logits + self.ap_prior(state)
        return (mean, log_std, ap_logits)

    def _continuous_sample(self, mean, log_std, deterministic: bool):
        std = log_std.exp()
        z = mean if deterministic else mean + std * torch.randn_like(std)
        action_cont = torch.tanh(z) * self.max_action
        log_prob_cont = (
            -0.5 * ((z - mean) / (std + 1e-08)) ** 2 - log_std - 0.5 * np.log(2 * np.pi)
        ).sum(dim=1, keepdim=True)
        squashed = torch.clamp(action_cont / max(self.max_action, 1e-08), -0.999999, 0.999999)
        log_prob_cont -= torch.sum(torch.log(1 - squashed.pow(2) + 1e-06), dim=1, keepdim=True)
        return (action_cont, log_prob_cont)

    def _ap_sample(self, ap_logits, deterministic: bool):
        probs = F.softmax(ap_logits, dim=-1)
        if deterministic:
            idx = torch.argmax(probs, dim=-1)
            ap_onehot = F.one_hot(idx, num_classes=self.num_rsu).to(dtype=ap_logits.dtype)
        else:
            tau = max(float(self.gumbel_tau), float(self.gumbel_tau_min))
            ap_onehot = F.gumbel_softmax(ap_logits, tau=tau, hard=True, dim=-1)
            idx = torch.argmax(ap_onehot, dim=-1)
        log_probs = F.log_softmax(ap_logits, dim=-1)
        selected_log_prob = torch.gather(log_probs, dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
        ap_log_prob_sum = selected_log_prob.sum(dim=1, keepdim=True)
        ap_log_prob_norm = selected_log_prob.mean(dim=1, keepdim=True)
        entropy = -(probs * torch.log(probs + 1e-08)).sum(dim=-1)
        max_prob = probs.max(dim=-1).values
        return (ap_onehot, ap_log_prob_sum, ap_log_prob_norm, entropy, max_prob)

    def sample(self, state: torch.Tensor, deterministic: bool = False, return_info: bool = False):
        mean, log_std, ap_logits = self.forward(state)
        action_cont, log_prob_cont = self._continuous_sample(mean, log_std, deterministic)
        ap_onehot, log_prob_ap, log_prob_ap_norm, ap_entropy, ap_max_prob = self._ap_sample(
            ap_logits, deterministic=deterministic
        )
        ap_flat = ap_onehot.reshape(state.shape[0], -1)
        action = torch.cat([action_cont, ap_flat], dim=1)
        total_log_prob = log_prob_cont + self.ap_entropy_weight * log_prob_ap_norm
        info = {
            "log_prob_cont": log_prob_cont,
            "log_prob_ap": log_prob_ap,
            "log_prob_ap_norm": log_prob_ap_norm,
            "ap_entropy": ap_entropy.mean(dim=1, keepdim=True),
            "ap_max_prob": ap_max_prob.mean(dim=1, keepdim=True),
            "ap_logits": ap_logits,
        }
        self.last_sample_info = {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in info.items()
            if key != "ap_logits"
        }
        if return_info:
            return (action, total_log_prob, info)
        return (action, total_log_prob)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 1)

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class CostCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.dropout = nn.Identity()
        self.out = nn.Linear(256, 1)

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        return self.out(x)
