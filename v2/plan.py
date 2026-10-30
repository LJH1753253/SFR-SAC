"""Experiment plan for the resubmission (main comparison, E1-E7, and ablations)."""
from __future__ import annotations

import copy
from pathlib import Path

from v2.runtime import HEURISTICS, LEARNERS, fingerprint
from v2.checkpoint import source_digest

# relative cost of one run (Safe SAC at N=70 = 1.0), used for scheduling and ETA
COST = {"safe": 1.0, "penalty_sac": 0.7, "vanilla": 0.7, "td3_lag": 0.5, "cpo": 0.2,
        "lyapunov": 0.03, "const_ratio": 0.03, "local_first": 0.002, "mb_greedy": 0.03}


def job_hash(cfg, method):
    """Hash of the parts of the configuration that affect this run only, so that changing one
    algorithm's settings does not invalidate the finished runs of the other methods."""
    sub = {k: cfg.get(k) for k in ("environment", "layout_seed_base", "cost_limit", "training", "method", "N",
                                   "seed", "probe_id", "evaluation_seed_base", "validation_seed_base", "intervention",
                                   "layout_seed", "evaluation_rng_seed")}
    sub["algorithm"] = cfg["algorithms"].get(method)
    sub["source_digest"] = source_digest()
    sub["execution"] = cfg.get("execution", {"device": "cpu"})
    if method in LEARNERS:
        sub["actor"] = cfg.get("actor")
    if method in ("const_ratio", "mb_greedy", "local_first"):
        sub["baselines"] = cfg.get("baselines")
    return fingerprint(sub)


def _job(root, cfg, kind, method, n, seed, experiment, variant, priority, frontier=False):
    cfg = copy.deepcopy(cfg)
    cfg.update(method=method, N=n, seed=seed)
    folder = Path(root) / "runs" / experiment / variant / f"N{n}" / f"seed{seed}"
    return {"kind": kind, "method": method, "N": n, "seed": seed, "experiment": experiment,
            "variant": variant, "priority": priority, "frontier": frontier, "config": cfg,
            "config_hash": job_hash(cfg, method), "directory": str(folder),
            "cost": COST[method] * (n / 70.0) * cfg["training"]["episodes"] / 1600.0}


def kind_of(method):
    return "learner" if method in LEARNERS else "lyapunov" if method == "lyapunov" else "heuristic"


def build_plan(cfg, root, main_n=(50, 55, 60, 65, 70), seeds=(0, 1, 2), abl_n=70,
               main_methods=LEARNERS + HEURISTICS, include=("main", "e1", "e2", "e3", "e5", "ablations")):
    jobs = []
    if "main" in include:
        for seed in seeds:
            for n in main_n:
                for m in main_methods:
                    jobs.append(_job(root, cfg, kind_of(m), m, n, seed, "main", m,
                                     1 if seed == seeds[0] else 2, frontier=(n == abl_n)))

    def variant(method, experiment, name, mutate, prio):
        for seed in seeds:
            c = copy.deepcopy(cfg)
            mutate(c)
            jobs.append(_job(root, c, kind_of(method), method, abl_n, seed, experiment, name, prio))

    def setd(v):
        return lambda c: c.__setitem__("cost_limit", v)

    if "e1" in include:  # R1.1: discounted (gamma = 0.99) instead of myopic critics, every learner
        for m in LEARNERS:
            variant(m, "e1_reward_horizon", f"{m}_gamma0.99",
                    lambda c: c["training"].__setitem__("gamma", 0.99), 3)
        variant("safe", "e1_reward_horizon", "safe_gamma0.99_discounted_cost",
                lambda c: (c["training"].__setitem__("gamma", 0.99),
                           c["algorithms"]["safe"].__setitem__("cost_critic_target_mode", "discounted_average")), 3)
    if "e2" in include:  # R1.2: untruncated (linear) penalty, budget sweep
        for d in (0.06, 0.08, 0.10, 0.12):
            def mut(c, d=d):
                c["algorithms"]["safe"]["penalty_mode"] = "linear"
                c["cost_limit"] = d
            variant("safe", "e2_penalty", f"linear_d{d:.2f}", mut, 3)
    if "e3" in include:  # R1.4: AP-independent gains and max-gain association
        variant("safe", "e3_ap", "ap_independent",
                lambda c: c["environment"].__setitem__("ap_dependent_gain", False), 3)
        variant("safe", "e3_ap", "max_gain_ap",
                lambda c: c["environment"].__setitem__("force_ap", "max_gain"), 3)
        for h in ("mb_greedy",):
            variant(h, "e3_ap", "ap_independent_" + h,
                    lambda c: c["environment"].__setitem__("ap_dependent_gain", False), 3)
    if "e5" in include:  # R1.6 / R2.C4: trade-off sweeps for matched-failure comparison
        for d in (0.06, 0.08, 0.12):
            variant("td3_lag", "e5_sweep", f"td3_d{d:.2f}", setd(d), 3)
            variant("cpo", "e5_sweep", f"cpo_d{d:.2f}", setd(d), 3)
        for beta in (1.0, 2.0, 10.0, 20.0, 50.0):
            variant("penalty_sac", "e5_sweep", f"penalty_beta{beta:g}",
                    lambda c, b=beta: c["algorithms"]["penalty_sac"].__setitem__("penalty_sac_beta", b), 3)
        for v in (0.5, 1.0, 5.0, 10.0):
            variant("lyapunov", "e5_sweep", f"lyapunov_V{v:g}",
                    lambda c, v=v: c["algorithms"]["lyapunov"].__setitem__("lyapunov_v_energy", v), 3)
    if "ablations" in include:
        for d in (0.06, 0.08, 0.09, 0.11, 0.12):
            variant("safe", "abl_budget", f"d{d:.2f}", setd(d), 4)
        for eta in (0.1, 0.2, 0.4, 0.5, 0.6):
            variant("safe", "abl_eta", f"eta{eta:.1f}",
                    lambda c, e=eta: c["environment"].__setitem__("ul_time_fraction", e), 4)
        for b0 in (0.0, 0.10):
            variant("safe", "abl_bgload", f"b0_{b0:.2f}",
                    lambda c, b=b0: c["environment"].__setitem__("mec_background_load_base", b), 4)
        variant("safe", "abl_actor", "no_ap_prior",
                lambda c: c.setdefault("actor", {}).__setitem__("ap_prior", False), 4)
        # same multiplier change per episode as the main setting (40 batch updates or 200 slots per episode)
        per_episode = 40 * cfg["algorithms"]["safe"]["lambda_lr"]
        for g, lr in (("episode", per_episode), ("step", per_episode / 200)):
            variant("safe", "abl_dual", g,
                    lambda c, g=g, lr=lr: c["algorithms"]["safe"].update(lambda_update_granularity=g, lambda_lr=lr), 4)
    jobs.sort(key=lambda j: (j["priority"], -j["cost"]))
    return jobs
