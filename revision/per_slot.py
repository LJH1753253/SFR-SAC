"""Direct per-slot optimizers: is the trained actor needed, or does optimizing the same model suffice?

With gamma = 0 the SFR-SAC actor amortizes a one-slot problem: minimize the critic's energy plus the
multiplier-weighted service-failure penalty. This module solves that problem directly in every slot.

* ``actor``           the trained SFR-SAC actor (deterministic), on the same test episodes.
* ``critic_refine``   start from the actor's action and descend the trained critics' objective
                      (the objective the actor was trained on), keeping the better of the two.
* ``critic_scratch``  the same optimization from a neutral start (best-channel AP, x = p = mid),
                      without the actor: what the learned model alone gives.
* ``oracle``          full-information reference: the simulator's exact slot model written in
                      PyTorch (true channels, CPU frequencies, energy factors, MEC capacity),
                      relaxed AP choice, multi-start (MB-greedy action and a neutral start), and an
                      online dual update of the failure multiplier toward the budget d.

AP choices are relaxed to a softmax during the descent and executed as one-hot argmax choices.
Decision times are measured per slot on one CPU thread.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from offloading.runtime import execution_action
from v2 import baselines as B
from v2.checkpoint import source_digest
from v2.runtime import evaluate_policy, load_agent, read_json, write_json

OPTIMIZERS = ("actor", "critic_refine", "critic_scratch", "oracle")


def _logit(p):
    p = torch.clamp(p, 1e-4, 1 - 1e-4)
    return torch.log(p) - torch.log1p(-p)


def _links(state, n, m):
    start = 3 * n + m + 8
    return state[:, start:start + n * m].reshape(-1, n, m)


class CriticOptimizer:
    """Gradient descent on the trained critics' one-slot objective."""

    def __init__(self, agent, iters, lr=0.05, start="actor"):
        self.agent, self.iters, self.lr, self.start = agent, int(iters), float(lr), start
        self.n, self.m = agent.num_devices, agent.num_rsu
        for module in (agent.actor, agent.critic_1, agent.critic_2, agent.cost_critic_1, agent.cost_critic_2):
            module.eval()
            module.requires_grad_(False)

    def objective(self, s, a):
        agent = self.agent
        q = torch.minimum(agent.critic_1(s, a), agent.critic_2(s, a))
        cost = torch.maximum(agent.cost_critic_1(s, a), agent.cost_critic_2(s, a)).clamp_min(0.0)
        violation = cost - agent.cost_limit
        if agent.penalty_mode == "hinge":
            violation = violation.clamp_min(0.0)
        return (-q + float(agent.lam.item()) * agent.cost_coef * violation).sum()

    def _action(self, u, logits, hard):
        ap = F.one_hot(logits.argmax(-1), self.m).to(u.dtype) if hard else F.softmax(logits, -1)
        return torch.cat([torch.tanh(u), ap.reshape(1, -1)], 1)

    def act(self, env, state):
        s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean, _, logits0 = self.agent.actor(s)
        if self.start == "actor":
            u, logits = mean.clone(), logits0.clone()
        else:
            u = torch.zeros_like(mean)
            links = _links(s, self.n, self.m)
            logits = 10.0 * (links - links.max(-1, keepdim=True).values)
        u.requires_grad_(True)
        logits.requires_grad_(True)
        optimizer = torch.optim.Adam([u, logits], lr=self.lr)
        for _ in range(self.iters):
            optimizer.zero_grad()
            self.objective(s, self._action(u, logits, hard=False)).backward()
            optimizer.step()
        with torch.no_grad():
            candidates = [self._action(u, logits, hard=True)]
            if self.start == "actor":
                candidates.append(self._action(mean, logits0, hard=True))
            best = min(candidates, key=lambda a: float(self.objective(s, a)))
        return execution_action(best.squeeze(0).numpy(), env, "safe")


class ExactSlotModel:
    """The simulator's slot outcome (v2.env.evaluate_slot) in PyTorch, with a relaxed AP choice."""

    def __init__(self, env):
        p = env.params
        if p.get("mec_sharing", "proportional") != "proportional":
            raise ValueError("The oracle models proportional MEC sharing only")
        self.env = env
        self.zeta, self.eff, self.pc = p["local_energy_zeta_bar"], p["tx_pa_efficiency"], p["tx_circuit_power"]
        self.airtime = p["tx_time_model"] == "tdma_airtime"

    def tensors(self):
        e = self.env
        t = lambda v: torch.as_tensor(np.asarray(v, dtype=np.float64))  # noqa: E731
        gain = t(10.0 ** (e._channel_db() / 10.0))
        band = t(e.rsu_access_bw)
        return dict(A=t(e.task_bits), cpb=t(e.task_cpb), cpu=t(e.local_cpu), factor=t(e.energy_factor),
                    gain=gain, band=band, noise=float(e.noise_psd), budget=float(e.mec_cpu * (1 - e.ul_time_fraction)
                                                                                 * e.slot_duration),
                    window=max(e.ul_time_fraction * e.slot_duration - e.backhaul_delay, 0.0),
                    fwd=float(e.backhaul_bw), D=float(e.slot_duration))

    def outcome(self, T, x, power, ap):
        """Mean energy (J per ED) and failure ratio; x, power: (B, N); ap: (B, N, M)."""
        off, local = x * T["A"], (1 - x) * T["A"]
        active = (off > 1e-6).to(x.dtype).detach()
        counts = (ap * active.unsqueeze(-1)).sum(1, keepdim=True)                     # (B, 1, M)
        share = (ap * counts.clamp_min(1.0)).sum(-1)                                  # (B, N)
        snr = power.unsqueeze(-1) * T["gain"] / (T["noise"] * T["band"])
        radio = (ap * T["band"] * torch.log2(1 + snr)).sum(-1)
        rate, fwd = radio / share, T["fwd"] / share
        cap = T["window"] / (1 / (rate + 1e-9) + 1 / (fwd + 1e-9)) * active
        y_del = torch.minimum(off, cap)
        y_loc = torch.minimum(local, T["cpu"] * T["D"] / T["cpb"])
        demand = (y_del * T["cpb"]).sum(1, keepdim=True)
        y_mec = y_del * torch.clamp(T["budget"] / demand.clamp_min(1e-12), max=1.0)
        cost = (T["A"] - y_loc - y_mec).sum(1) / T["A"].sum()
        e_loc = self.zeta * T["factor"] * T["cpu"] ** 2 * T["cpb"] * y_loc
        airtime = y_del / ((radio if self.airtime else rate) + 1e-9)
        e_tx = T["factor"] * (power / self.eff + self.pc) * airtime
        return (e_loc + e_tx).mean(1), cost


class OracleOptimizer:
    """Full-information per-slot primal-dual optimizer (reference, not deployable)."""

    def __init__(self, cost_limit, iters, lr=0.2, dual_lr=0.5, mu0=3.0, mb_slack=1.0):
        self.d, self.iters, self.lr = float(cost_limit), int(iters), float(lr)
        self.dual_lr, self.mu = float(dual_lr), float(mu0)
        self.mb = B.MBGreedy(slack=mb_slack)

    def post(self, info):
        self.mu = max(0.0, self.mu + self.dual_lr * (float(info["cost"]) - self.d))

    def _vector(self, env, x, p01, ap_index):
        onehot = np.eye(env.R, dtype=np.float32)[ap_index].ravel()
        return np.concatenate([2 * x - 1, 2 * p01 - 1, 2 * onehot - 1]).astype(np.float32)

    def act(self, env, state):
        model = ExactSlotModel(env)
        T = model.tensors()
        m = env.R
        pmin, pmax = env.tx_power_min, env.tx_power_max
        mb_action = np.asarray(self.mb.act(env), dtype=np.float32)
        x0, p0, ap0 = env.decode_action(mb_action)
        # Two starts optimized as one batch: the MB-greedy decision and a neutral best-channel start.
        x_init = np.stack([x0, np.full_like(x0, .5)])
        p_init = np.stack([(p0 - pmin) / (pmax - pmin), np.full_like(x0, .5)])
        ap_init = np.stack([ap0, np.argmax(env._channel_db(), 1)])
        ux = _logit(torch.as_tensor(x_init)).clone().requires_grad_(True)
        up = _logit(torch.as_tensor(p_init)).clone().requires_grad_(True)
        logits = (4.0 * F.one_hot(torch.as_tensor(ap_init).long(), m).double()).requires_grad_(True)
        optimizer = torch.optim.Adam([ux, up, logits], lr=self.lr)
        for k in range(self.iters):
            tau = 1.0 - 0.8 * k / max(self.iters - 1, 1)
            energy, cost = model.outcome(T, torch.sigmoid(ux), pmin + torch.sigmoid(up) * (pmax - pmin),
                                         F.softmax(logits / tau, -1))
            loss = (energy / env.energy_ref + self.mu * cost).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        candidates = [mb_action]
        with torch.no_grad():
            for b in range(2):
                candidates.append(self._vector(env, torch.sigmoid(ux[b]).numpy(), torch.sigmoid(up[b]).numpy(),
                                               logits[b].argmax(-1).numpy()))

        def exact_objective(vector):
            out = env.evaluate_slot(*env.decode_action(vector))
            return out["energy"] / env.energy_ref + self.mu * out["cost"]
        return min(candidates, key=exact_objective)


class Timed:
    def __init__(self, act):
        self.act, self.seconds = act, []

    def __call__(self, env, state):
        start = time.perf_counter()
        action = self.act(env, state)
        self.seconds.append(time.perf_counter() - start)
        return action


def run(spec, out):
    """Evaluate one optimizer on the declared test episodes (CPU, one thread)."""
    out = Path(out)
    torch.set_num_threads(1)
    torch.set_flush_denormal(True)
    parent = Path(spec["parent"])
    cfg = read_json(parent / "job.json")["config"]
    cfg = dict(cfg, execution=dict(device="cpu", require_cuda=False, tf32=False))
    source = read_json(parent / "final_eval_summary.json")
    if source["source_digest"] != source_digest():
        raise RuntimeError("Parent model was trained with different source code")
    n, seed, kind = spec["N"], spec["seed"], spec["optimizer"]
    agent, method, _, _ = load_agent(parent / "model.pt", read_json(parent / "job.json")["config"], device="cpu")
    post, extra = None, {}
    if kind == "actor":
        policy = Timed(lambda env, state: execution_action(agent.select_action(state, deterministic=True), env, method))
    elif kind in {"critic_refine", "critic_scratch"}:
        optimizer = CriticOptimizer(agent, spec["iters"], spec.get("lr", 0.05),
                                    start="actor" if kind == "critic_refine" else "neutral")
        policy = Timed(optimizer.act)
    elif kind == "oracle":
        optimizer = OracleOptimizer(cfg["cost_limit"], spec["iters"], dual_lr=spec.get("dual_lr", 0.5),
                                    mu0=spec.get("mu0", 3.0))
        # Adapt the multiplier on separate warm-up episodes before the test episodes.
        evaluate_policy(cfg, n, seed, optimizer.act, spec["warmup_episodes"], spec["warmup_rng_seed"],
                        post_step=optimizer.post)
        extra["oracle_mu_after_warmup"] = optimizer.mu
        policy, post = Timed(optimizer.act), optimizer.post
    else:
        raise ValueError(f"Unknown optimizer {kind}")
    summary = evaluate_policy(cfg, n, seed, policy, spec["test_episodes"], spec["test_rng_seed"],
                              post_step=post, details_dir=out)
    seconds = np.asarray(policy.seconds)
    summary.update(extra, optimizer=kind, iterations=spec.get("iters", 0),
                   decision_ms_median=float(1000 * np.median(seconds)),
                   decision_ms_p95=float(1000 * np.percentile(seconds, 95)),
                   decision_ms_mean=float(1000 * seconds.mean()))
    if kind == "oracle":
        summary["oracle_mu_final"] = optimizer.mu
    write_json(out / "test_summary.json", dict(id=spec["id"], source_digest=source_digest(),
                                               spec={k: v for k, v in spec.items() if k != "config"},
                                               summary=summary, split="held_out_test",
                                               test_rng_seed=spec["test_rng_seed"]))
    return summary


__all__ = ["OPTIMIZERS", "CriticOptimizer", "OracleOptimizer", "ExactSlotModel", "run", "math"]
