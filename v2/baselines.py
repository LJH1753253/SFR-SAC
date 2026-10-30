"""Non-learning reference policies for the v2 environment.

All policies return an action in the environment's format
[offload (N) in [-1,1], power (N) in [-1,1], AP block (N*M)], so they are evaluated with
exactly the same environment step as the learned policies.

const_ratio  : one offloading ratio x and one transmit power for all EDs, AP with the best
               current channel. Needs no model. Knobs: x, p.
local_first  : keep each ED's workload local up to its CPU capacity, offload only the
               overflow at p_max via the best-channel AP. Needs A_i, kappa_i, f_i.
mb_greedy    : model-based per-slot greedy (full knowledge of channels, CPUs, energy factors,
               MEC load): mandatory overflow first with the lowest sufficient power,
               load-aware AP assignment, then extra offloading while it saves device energy
               and fits an MEC cycle budget s * f_mec * (1-eta) * D. Knob: s.
"""
from __future__ import annotations

import numpy as np

from agents.lyapunov_dpp_agent import LyapunovDPPAgent


def to_action(env, x, p, ap):
    a_off = 2.0 * np.clip(x, 0.0, 1.0) - 1.0
    a_pow = 2.0 * (np.asarray(p, float) - env.tx_power_min) / (env.tx_power_max - env.tx_power_min) - 1.0
    block = np.full((env.N, env.R), -1.0)
    block[np.arange(env.N), ap] = 1.0
    return np.concatenate([a_off, np.clip(a_pow, -1, 1), block.ravel()]).astype(np.float32)


def best_ap(env):
    return np.argmax(env._channel_db(), axis=1)


class ConstRatio:
    def __init__(self, x=0.3, p=None):
        self.x, self.p = float(x), p

    def act(self, env):
        p = env.tx_power_max if self.p is None else float(self.p)
        return to_action(env, np.full(env.N, self.x), np.full(env.N, p), best_ap(env))


class LocalFirst:
    def act(self, env):
        A = env.task_bits
        over = np.maximum(A - env.local_capacity_bits(), 0.0)
        return to_action(env, over / A, np.full(env.N, env.tx_power_max), best_ap(env))


class MBGreedy:
    def __init__(self, slack=1.0, power_levels=8):
        self.slack = float(slack)
        self.levels = int(power_levels)

    def act(self, env):
        N, M = env.N, env.R
        A = env.task_bits
        cpb = env.task_cpb
        D = env.slot_duration
        window = max(env.ul_time_fraction * D - env.backhaul_delay, 1e-6)
        g_lin = 10.0 ** (env._channel_db() / 10.0)                     # (N, M)
        over = np.maximum(A - env.local_capacity_bits(), 0.0)
        zeta = env.params["local_energy_zeta_bar"]
        e_loc_bit = zeta * env.energy_factor * env.local_cpu**2 * cpb   # J/bit if computed locally
        powers = np.linspace(env.tx_power_min, env.tx_power_max, self.levels)
        # --- load-aware AP assignment for EDs that will offload something
        local_cost_high = e_loc_bit > 0  # all EDs are candidates; mandatory ones first
        order = np.argsort(-(over + 1e-3 * A * local_cost_high))
        counts = np.zeros(M)
        ap = np.argmax(g_lin, axis=1)
        for i in order:
            snr = env.tx_power_max * g_lin[i] / (env.noise_psd * env.rsu_access_bw)
            r = env.rsu_access_bw / (counts + 1.0) * np.log2(1.0 + snr)
            ap[i] = int(np.argmax(r))
            counts[ap[i]] += 1.0
        band = env.rsu_access_bw[ap]
        gi = g_lin[np.arange(N), ap]
        radio_rate = band[:, None] * np.log2(1.0 + powers[None, :] * gi[:, None] / (env.noise_psd * band[:, None]))
        rate = radio_rate / np.maximum(counts[ap, None], 1.0)
        forwarding = env.backhaul_bw / np.maximum(counts[ap, None], 1.0)
        cap = window / (1.0 / (rate + 1e-9) + 1.0 / (forwarding + 1e-9))
        draw = env.energy_factor[:, None] * (powers[None, :] / env.params["tx_pa_efficiency"]
                                             + env.params["tx_circuit_power"])
        energy_rate = radio_rate if env.params["tx_time_model"] == "tdma_airtime" else rate
        e_tx_bit = draw / (energy_rate + 1e-9)
        off = np.zeros(N)
        pidx = np.full(N, self.levels - 1)
        # --- mandatory overflow with the cheapest sufficient power
        for i in np.flatnonzero(over > 0):
            ok = np.flatnonzero(cap[i] >= over[i])
            pidx[i] = ok[np.argmin(e_tx_bit[i, ok])] if ok.size else self.levels - 1
            off[i] = min(over[i], cap[i, pidx[i]])
        budget = self.slack * env.mec_cpu * (1.0 - env.ul_time_fraction) * D - np.sum(off * cpb)
        # --- extra offloading where it saves device energy, best saving per MEC cycle first
        saving = (e_loc_bit[:, None] - e_tx_bit).max(axis=1) / cpb
        for i in np.argsort(-saving):
            if budget <= 0 or saving[i] <= 0:
                break
            extra = min(A[i] - off[i] - max(A[i] - off[i] - env.local_capacity_bits()[i], 0.0),
                        budget / cpb[i])
            if extra <= 0:
                continue
            target = off[i] + extra
            ok = np.flatnonzero(cap[i] >= target)
            j = ok[np.argmin(e_tx_bit[i, ok])] if ok.size else self.levels - 1
            target = min(target, cap[i, j])
            if (e_loc_bit[i] - e_tx_bit[i, j]) <= 0 or target <= off[i]:
                continue
            budget -= (target - off[i]) * cpb[i]
            off[i], pidx[i] = target, j
        x = np.where(A > 0, off / A, 0.0)
        return to_action(env, x, powers[pidx], ap)


def rollout(env, policy, episodes, seed, budget, collect_device=False):
    """Evaluate a baseline; returns summary dict (same keys as learned-policy evaluation)."""
    from v2.metrics import EvalAccumulator
    np.random.seed(seed)
    acc = EvalAccumulator(env.N, budget)
    for _ in range(episodes):
        env.reset()
        done = False
        while not done:
            _, _, done, info = env.step(policy.act(env))
            acc.add(info)
        acc.end_episode()
    return acc.summary()


class LyapunovV2(LyapunovDPPAgent):
    """Lyapunov-inspired discrete rule with a rate-aware AP association.

    The archived rule assigns active EDs to the least-loaded AP, which ignores the AP-dependent
    channels of the v2 environment. Here active EDs (heaviest workload first) pick the AP that
    maximizes their time-shared spectral efficiency log2(1 + SNR_im) / (n_m + 1).
    """

    def __init__(self, env, **kwargs):
        super().__init__(**kwargs)
        self.link_offset = int(env.link_state_offset)

    def select_action(self, state, deterministic=False):
        a = np.array(super().select_action(state, deterministic), dtype=np.float32)
        n, r = self.N, self.R
        s = np.asarray(state, dtype=np.float64).ravel()
        q = s[self.link_offset : self.link_offset + n * r].reshape(n, r)
        se = np.log2(1.0 + 10.0 ** ((50.0 * q - 10.0) / 10.0))
        idx = np.argmax(q, axis=1)
        active = np.flatnonzero((a[:n] + 1.0) / 2.0 > 1e-6)
        load = np.zeros(r)
        for i in active[np.argsort(-s[n : 2 * n][active], kind="stable")]:
            m = int(np.argmax(se[i] / (load + 1.0)))
            idx[i] = m
            load[m] += 1.0
        a[2 * n : 3 * n] = self._ap_indices_to_action(idx)
        return a
