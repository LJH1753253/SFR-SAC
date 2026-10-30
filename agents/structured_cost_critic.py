"""Immediate service-failure predictor with a learned edge-delivery stage."""

from __future__ import annotations

import torch
from torch import nn


class ServiceFlowCostCritic(nn.Module):
    def __init__(self, state_dim, action_dim, *, num_devices, num_rsu, parameters, hidden=64):
        super().__init__()
        self.n, self.m = int(num_devices), int(num_rsu)
        if state_dim != 4 * self.n + self.m + 8 + self.n * self.m:
            raise ValueError("Service-flow critic requires the v2 observation layout")
        if action_dim != 2 * self.n + self.n * self.m:
            raise ValueError("Service-flow critic requires hybrid one-hot actions")
        if parameters["force_ap"] != "none":
            raise ValueError("Service-flow critic expects the actor's AP choice to be executed")
        self.p = dict(parameters)
        self.link_start = 3 * self.n + self.m + 8
        self.delivery_net = nn.Sequential(
            nn.Linear(18, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def decode_state(self, state):
        p, n = self.p, self.n
        arrivals = state[:, n:2 * n] * p["arrival_high"] * p["load_scale"] * p["burst_multiplier"]
        cpb = p["cycles_per_bit"] * (
            p["cycles_low_factor"]
            + state[:, 2 * n:3 * n] * (p["cycles_high_factor"] - p["cycles_low_factor"])
        )
        cpu = p["local_cpu_min"] + state[:, -n:] * (p["local_cpu_max"] - p["local_cpu_min"])
        available = state[:, 3 * n + self.m:3 * n + self.m + 1]
        bg = p["mec_background_load_max"] - available * (
            p["mec_background_load_max"] - p["mec_background_load_min"]
        )
        budget = p["bs_cpu_base"] * (1.0 - bg) * (1.0 - p["ul_time_fraction"]) * p["slot_duration"]
        links = state[:, self.link_start:self.link_start + n * self.m].reshape(-1, n, self.m)
        return arrivals, cpb, cpu, budget, links

    def compose_cost(self, arrivals, cpb, local_success, delivered, budget):
        demand = (cpb * delivered).sum(dim=1, keepdim=True)
        ratio = torch.clamp(budget / demand.clamp_min(1e-12), max=1.0)
        mec_success = delivered * ratio
        cost = (arrivals - local_success - mec_success).sum(dim=1, keepdim=True)
        return torch.clamp(cost / arrivals.sum(dim=1, keepdim=True).clamp_min(1e-12), 0.0, 1.0)

    def features(self, state, action):
        p, n = self.p, self.n
        arrivals, cpb, cpu, budget, links = self.decode_state(state)
        x = torch.clamp((action[:, :n] + 1.0) * 0.5, 0.0, 1.0)
        power_proxy = torch.clamp((action[:, n:2 * n] + 1.0) * 0.5, 0.0, 1.0)
        power = p["tx_power_min"] + power_proxy * (p["tx_power_max"] - p["tx_power_min"])
        # Preserve ST gradients at the zero entries of a hard one-hot action.
        ap = action[:, 2 * n:].reshape(-1, n, self.m)
        off = x * arrivals
        active = (off > 1e-6).to(state.dtype)
        selected_link = (links * ap).sum(dim=-1)
        counts = (ap * active.unsqueeze(-1)).sum(dim=1)
        ap_bits = torch.einsum("bnm,bn->bm", ap, off)
        ap_cycles = torch.einsum("bnm,bn->bm", ap, off * cpb)
        ap_quality = torch.einsum("bnm,bn->bm", ap, selected_link * active) / counts.clamp_min(1.0)

        def selected(values):
            return (ap * values.unsqueeze(1)).sum(dim=-1)

        total_bits = arrivals.sum(dim=1, keepdim=True).clamp_min(1e-12)
        local_cap = cpu * p["slot_duration"] / cpb
        local_success = torch.minimum((1.0 - x) * arrivals, local_cap)
        global_off = off.sum(dim=1, keepdim=True) / total_bits
        global_cycles = (off * cpb).sum(dim=1, keepdim=True) / budget.clamp_min(1e-12)
        mec_fraction = budget / (p["bs_cpu_base"] * (1.0 - p["ul_time_fraction"]) * p["slot_duration"])
        feature = torch.stack([
            state[:, n:2 * n], cpb / p["cycles_per_bit"], state[:, -n:], x, power_proxy,
            selected_link, links.max(dim=-1).values, torch.log(power / p["tx_power_max"]),
            selected(counts) / n, selected(ap_bits) / total_bits,
            selected(ap_cycles) / budget.clamp_min(1e-12), selected(ap_quality),
            mec_fraction.expand_as(x), torch.clamp(local_cap / arrivals.clamp_min(1e-12), max=2.0),
            global_off.expand_as(x), global_cycles.expand_as(x),
            torch.full_like(x, p["ul_time_fraction"]),
            torch.full_like(x, p["backhaul_delay"] / p["slot_duration"]),
        ], dim=-1)
        return feature, arrivals, cpb, cpu, budget, local_success, x, power, counts, ap

    def predict(self, state, action):
        feature, arrivals, cpb, _, budget, local_success, x, *_ = self.features(state, action)
        delivery_fraction = x * torch.sigmoid(self.delivery_net(feature).squeeze(-1))
        cost = self.compose_cost(arrivals, cpb, local_success, delivery_fraction * arrivals, budget)
        return cost, delivery_fraction

    def forward(self, state, action):
        return self.predict(state, action)[0]
