"""Hybrid policy with shared device features and pooled resource context."""

import torch
from torch import nn

from agents.networks import HybridActor, make_ap_prior


class ServiceFlowActor(HybridActor):
    def __init__(self, state_dim, num_devices, num_rsu, *, parameters, hidden=64,
                 embedding_dim=8, max_action=1.0, gumbel_tau=1.0,
                 gumbel_tau_min=0.3, ap_entropy_weight=1.0, ap_prior=None,
                 use_physics_features=True, use_pooled_context=True, use_device_embedding=True):
        nn.Module.__init__(self)
        self.num_devices, self.num_rsu = int(num_devices), int(num_rsu)
        n, m = self.num_devices, self.num_rsu
        if state_dim != 4*n + m + 8 + n*m:
            raise ValueError("Service-flow actor requires the v2 observation layout")
        self.p = dict(parameters)
        self.continuous_dim, self.ap_dim = 2*n, n*m
        self.action_dim = self.continuous_dim + self.ap_dim
        self.max_action = float(max_action)
        self.gumbel_tau, self.gumbel_tau_min = float(gumbel_tau), float(gumbel_tau_min)
        self.ap_entropy_weight = float(ap_entropy_weight)
        self.LOG_STD_MIN, self.LOG_STD_MAX = -20, 2
        self.last_sample_info = {}
        self.use_physics_features = bool(use_physics_features)
        self.use_pooled_context = bool(use_pooled_context)
        self.use_device_embedding = bool(use_device_embedding)
        self.device_embedding = nn.Embedding(n, embedding_dim)
        nn.init.normal_(self.device_embedding.weight, std=.02)
        self.device_encoder = nn.Sequential(
            nn.Linear(8 + m + embedding_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.global_encoder = nn.Sequential(
            nn.Linear(2*hidden + m + 8 + 3, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.continuous_head = nn.Sequential(nn.Linear(2*hidden, hidden), nn.ReLU(),
                                             nn.Linear(hidden, 4))
        self.ap_head = nn.Sequential(nn.Linear(2*hidden + 4, hidden//2), nn.ReLU(),
                                     nn.Linear(hidden//2, 1))
        self.ap_prior = make_ap_prior(ap_prior, n, m)

    def observed_features(self, state):
        n, m, p = self.num_devices, self.num_rsu, self.p
        arrivals_norm = state[:, n:2*n]
        arrivals = arrivals_norm * p["arrival_high"] * p["load_scale"] * p["burst_multiplier"]
        cpb = p["cycles_per_bit"] * (p["cycles_low_factor"] + state[:, 2*n:3*n] *
                                    (p["cycles_high_factor"] - p["cycles_low_factor"]))
        cpu_norm = state[:, -n:]
        cpu = p["local_cpu_min"] + cpu_norm * (p["local_cpu_max"] - p["local_cpu_min"])
        local_capacity = cpu * p["slot_duration"] / cpb
        capacity_ratio = local_capacity / arrivals.clamp_min(1e-6)
        # This uses a unit energy factor, not the simulator's hidden device coefficients.
        nominal_energy = p["local_energy_zeta_bar"] * cpu.square() * cpb * arrivals / p["energy_ref"]
        link_start = 3*n + m + 8
        links = state[:, link_start:link_start+n*m].reshape(-1, n, m)
        per_device = torch.stack((arrivals_norm, cpb/p["cycles_per_bit"], cpu_norm,
                                  capacity_ratio.clamp(max=4)/4, nominal_energy.clamp(max=4)/4,
                                  state[:, :n], links.mean(-1), links.std(-1, unbiased=False)), -1)
        previous_load = state[:, 3*n:3*n+m]
        scalars = state[:, 3*n+m:3*n+m+8]
        bg = p["mec_background_load_max"] - scalars[:, :1] * (
            p["mec_background_load_max"] - p["mec_background_load_min"])
        mec_budget = p["bs_cpu_base"] * (1-bg) * (1-p["ul_time_fraction"]) * p["slot_duration"]
        resource = torch.cat(((arrivals*cpb).sum(1, keepdim=True)/mec_budget.clamp_min(1e-6),
                               local_capacity.sum(1, keepdim=True)/arrivals.sum(1, keepdim=True).clamp_min(1e-6),
                               nominal_energy.mean(1, keepdim=True)), 1).clamp(max=8)/8
        return per_device, links, previous_load, scalars, resource

    def forward(self, state):
        n, m = self.num_devices, self.num_rsu
        features, links, previous_load, scalars, resource = self.observed_features(state)
        if not self.use_physics_features:
            features = features.clone()
            features[:, :, 3:5] = 0
            resource = torch.zeros_like(resource)
        identity = self.device_embedding.weight.unsqueeze(0).expand(state.shape[0], -1, -1)
        if not self.use_device_embedding:
            identity = torch.zeros_like(identity)
        tokens = self.device_encoder(torch.cat((features, links, identity), -1))
        pooled = torch.cat((tokens.mean(1), tokens.max(1).values, previous_load, scalars, resource), 1)
        global_token = self.global_encoder(pooled).unsqueeze(1).expand(-1, n, -1)
        if not self.use_pooled_context:
            global_token = torch.zeros_like(global_token)
        context = torch.cat((tokens, global_token), -1)
        continuous = self.continuous_head(context)
        mean = torch.cat((continuous[:, :, 0], continuous[:, :, 1]), 1)
        log_std = torch.cat((continuous[:, :, 2], continuous[:, :, 3]), 1).clamp(
            self.LOG_STD_MIN, self.LOG_STD_MAX)
        pair = torch.stack((links, links-links.max(-1, keepdim=True).values,
                            previous_load.unsqueeze(1).expand(-1, n, -1),
                            links.mean(1, keepdim=True).expand(-1, n, -1)), -1)
        ap_logits = self.ap_head(torch.cat((context.unsqueeze(2).expand(-1, -1, m, -1), pair), -1)).squeeze(-1)
        if self.ap_prior is not None:
            ap_logits = ap_logits + self.ap_prior(state)
        return mean, log_std, ap_logits


class PhysicalFeatureMLPActor(HybridActor):
    """The original dense policy augmented with the same derived physical features."""

    def __init__(self, state_dim, num_devices, num_rsu, *, parameters, **kwargs):
        super().__init__(state_dim + 8*num_devices + 3, num_devices, num_rsu, **kwargs)
        self.p = dict(parameters)

    def forward(self, state):
        features, _, _, _, resource = ServiceFlowActor.observed_features(self, state)
        augmented = torch.cat((state, features.reshape(state.shape[0], -1), resource), 1)
        return super().forward(augmented)
