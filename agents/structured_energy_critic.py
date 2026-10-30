"""Immediate Joule-based energy predictor; no action or constraint-response wrapper."""

import torch
import torch.nn.functional as F
from torch import nn

from agents.structured_cost_critic import ServiceFlowCostCritic


class ServiceFlowEnergyCritic(ServiceFlowCostCritic):
    def __init__(self, *args, reward_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if self.p["tx_time_model"] != "tdma_airtime":
            raise ValueError("Structured energy critic requires the TDMA-airtime energy model")
        self.reward_scale = float(reward_scale)
        # Fixed deployment coefficients are learned from outcomes, not read from the environment.
        self.log_energy_factor = nn.Parameter(torch.zeros(self.n))

    def predict_energy(self, state, action):
        feature, arrivals, cpb, cpu, _, local_success, x, power, counts, ap = self.features(state, action)
        factor = torch.exp(self.log_energy_factor.clamp(-2.0, 2.0)).unsqueeze(0)
        local = self.p["local_energy_zeta_bar"] * factor * cpu.square() * cpb * local_success
        seconds_per_bit = 1e-7 * F.softplus(self.delivery_net(feature).squeeze(-1))
        share = (ap * counts.unsqueeze(1)).sum(dim=-1).clamp_min(1.0)
        window = max(self.p["ul_time_fraction"] * self.p["slot_duration"] - self.p["backhaul_delay"], 0.0)
        airtime = torch.minimum(x * arrivals * seconds_per_bit, window / share)
        tx = factor * (power / self.p["tx_pa_efficiency"] + self.p["tx_circuit_power"]) * airtime
        components = torch.stack((local, tx), dim=-1) / self.p["energy_ref"]
        reward = -self.reward_scale * components.sum(dim=-1).mean(dim=1, keepdim=True)
        return reward, components

    def forward(self, state, action):
        return self.predict_energy(state, action)[0]
