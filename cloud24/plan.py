"""Declared training arms, held-out streams, and validation-only selection."""

import copy
from pathlib import Path

import numpy as np

from v2.checkpoint import source_digest
from v2.env import DEFAULTS
from v2.plan import job_hash
from v2.runtime import fingerprint, read_json

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ("shared_exact", "split_exact", "split_bounded", "split_values")
PHASES = ("screen", "confirm", "main", "ablations", "tradeoff", "extension")
# Cumulative wall-clock ceilings, excluding the benchmark. Export reserve: 90 minutes.
PHASE_FRACTIONS = (.23, .40, .77, .87, .94, 1.)


def base_config():
    cfg = read_json(ROOT / "configs/base.json")
    defaults = {k: v for k, v in DEFAULTS.items() if k not in {"num_devices", "layout_seed"}}
    cfg["environment"] = {**defaults, **cfg["environment"]}
    return cfg


def configure(variant, *, n=70, layout=1000, train_seed=0, selected="shared_exact", small=False):
    cfg = base_config()
    cfg.update(layout_seed=int(layout), seed=int(train_seed), N=int(n))
    cfg["evaluation_rng_seed"] = 700000 + 100*n + (layout-1000)
    cfg["validation_seed_base"] = 600000 + 100*(layout-1000)
    method = "safe"
    architecture = selected if variant not in CANDIDATES else variant
    if variant in {"sac_lagrangian", "td3", "dpp_rule", "constant", "local_first", "mb_greedy"}:
        architecture = "shared_exact"
    if variant == "split_fixed_prior":
        architecture = "split_exact"
    safe = cfg["algorithms"]["safe"]
    if architecture != "shared_exact":
        safe["actor_architecture"] = "split_service_flow"
    if architecture in {"split_bounded", "split_values"}:
        safe["ap_residual_bound"] = 2.0
        cfg["actor"]["ap_prior_learnable"] = False
    if architecture == "split_values":
        safe.update(ap_gradient_mode="discrete_values", ap_value_devices=4, ap_value_chunk=512)
    if variant == "sac_lagrangian":
        safe.update(cost_critic_architecture="mlp", reward_critic_architecture="mlp", penalty_mode="linear")
    elif variant.startswith("penalty") or variant == "vanilla":
        method = "vanilla" if variant == "vanilla" else "penalty_sac"
        for key in ("actor_architecture", "actor_hidden", "actor_embedding_dim", "ap_residual_bound"):
            if key in safe:
                cfg["algorithms"][method][key] = safe[key]
        if variant.startswith("penalty_beta"):
            cfg["algorithms"][method]["penalty_sac_beta"] = float(variant.split("beta")[1])
    elif variant == "td3":
        method = "td3_lag"
    elif variant in {"dpp_rule", "constant", "local_first", "mb_greedy"}:
        method = {"dpp_rule": "lyapunov", "constant": "const_ratio"}.get(variant, variant)
        cfg["training"].update(episodes=20, eval_episodes=100)
        cfg["execution"] = dict(device="cpu", require_cuda=False, tf32=False)
    elif variant == "no_pooled_context":
        safe["actor_use_pooled_context"] = False
    elif variant == "mlp_critics":
        safe.update(cost_critic_architecture="mlp", reward_critic_architecture="mlp")
    elif variant == "st_gradient":
        safe["ap_gradient_mode"] = "straight_through"
    elif variant == "split_fixed_prior":
        cfg["actor"]["ap_prior_learnable"] = False
    elif variant.startswith("safe_d"):
        cfg["cost_limit"] = float(variant.split("safe_d")[1])
    elif variant not in CANDIDATES:
        raise ValueError(f"Unknown variant: {variant}")
    cfg["method"] = method
    if small:
        cfg["environment"]["max_ep_steps"] = 12
        cfg["training"].update(episodes=2, eval_episodes=1, warmup_steps=16,
                               batch_size=8, update_freq=2, replay_capacity=128, checkpoint_every=1)
        cfg["baselines"].update(validation_episodes=1, const_x_grid=[.3],
                                const_p_grid=[.01], mb_slack_grid=[1.])
    return cfg


def training_job(output, variant, *, n=70, layout=1000, seed=0, selected="shared_exact", small=False):
    cfg = configure(variant, n=n, layout=layout, train_seed=seed, selected=selected, small=small)
    method = cfg["method"]
    identity = job_hash(cfg, method)
    # Phase-independent identities permit exact reuse without reclassifying validation as test.
    folder = Path(output)/"runs"/f"N{n}"/f"layout{layout}"/f"train{seed}"/identity
    return dict(id=identity, task="train", kind="learner" if method in {"safe", "vanilla", "penalty_sac", "td3_lag"}
                else "lyapunov" if method == "lyapunov" else "heuristic",
                method=method, N=n, seed=seed, layout_seed=layout, variant=variant,
                experiment="cloud24", config=cfg, config_hash=identity, directory=str(folder.resolve()),
                resource="cpu" if cfg["execution"]["device"] == "cpu" else "gpu")


def test_job(output, parent, variant, small=False):
    path = Path(output)/"test"/variant/f"N{parent['N']}"/f"layout{parent['layout_seed']}"/f"train{parent['seed']}"
    spec = {k: parent[k] for k in ("method", "N", "seed", "layout_seed", "config_hash", "config")}
    spec.update(task="test", variant=variant, directory=str(path.resolve()),
                parent=parent["directory"], test_episodes=1 if small else 100,
                test_rng_seed=900000+parent["N"]*100+(parent["layout_seed"]-1000), resource=parent["resource"])
    spec["id"] = fingerprint({k: v for k, v in spec.items() if k != "directory"})
    return spec


def screening(output, small=False):
    return [training_job(output, variant, layout=1000+i, seed=i, small=small)
            for i in range(3) for variant in CANDIDATES]


def confirmation(output, candidate, small=False):
    return [training_job(output, v, layout=1000+layout, seed=seed, small=small)
            for seed in range(3) for layout in range(3) if seed != layout
            for v in ("shared_exact", candidate)] if candidate != "shared_exact" else []


def main_tasks(output, selected, small=False, extension=False):
    ns = (55, 65) if extension else (70, 50, 60)
    variants = (selected, "sac_lagrangian", "penalty_beta5", "vanilla", "td3",
                "constant", "local_first", "mb_greedy", "dpp_rule")
    tasks = []
    for seed in range(5):
        for n in ns:
            for variant in variants:
                parent = training_job(output, variant, n=n, layout=1000+seed, seed=seed,
                                      selected=selected, small=small)
                tasks.extend((parent, test_job(output, parent, variant, small)))
    return tasks


def ablation_tasks(output, selected, small=False):
    variants = ["shared_exact", "split_exact", "split_fixed_prior", "split_bounded", "no_pooled_context", "mlp_critics"]
    if selected == "split_values":
        variants.append("st_gradient")
    tasks = []
    for seed in range(3):
        for variant in variants:
            parent = training_job(output, variant, layout=1000+seed, seed=seed,
                                  selected=selected, small=small)
            tasks.extend((parent, test_job(output, parent, variant, small)))
    return tasks


def tradeoff_tasks(output, selected, small=False):
    tasks = []
    for seed in range(3):
        for variant in ("safe_d0.08", "safe_d0.12", "penalty_beta1", "penalty_beta2", "penalty_beta10", "penalty_beta20"):
            parent = training_job(output, variant, layout=1000+seed, seed=seed,
                                  selected=selected, small=small)
            tasks.extend((parent, test_job(output, parent, variant, small)))
    return tasks


def completed_summary(job):
    path = Path(job["directory"])/"final_eval_summary.json"
    if not path.exists():
        return None
    result = read_json(path)
    if result.get("config_hash") != job["config_hash"] or result.get("source_digest") != source_digest():
        raise RuntimeError(f"Existing result identity mismatch: {path}")
    return result["summary"]


def select_screen(jobs):
    metrics = {}
    for variant in CANDIDATES:
        summaries = [completed_summary(j) for j in jobs if j["variant"] == variant]
        if len(summaries) != 3 or any(s is None for s in summaries):
            continue
        metrics[variant] = {key: float(np.mean([s[key] for s in summaries]))
                            for key in ("cost_percent", "energy_mJ_device_slot", "bfsr_percent")}
        metrics[variant]["worst_cost_percent"] = max(s["cost_percent"] for s in summaries)
    baseline = metrics.get("shared_exact")
    eligible = [v for v in CANDIDATES[1:] if v in metrics and baseline is not None
                and metrics[v]["cost_percent"] <= 10.5 and metrics[v]["worst_cost_percent"] <= 11.5]
    if not eligible:
        return dict(candidate="shared_exact", complete=False, metrics=metrics,
                    reason="No complete non-reference candidate meets the screening guardrails.")
    def score(v):
        row = metrics[v]
        return (4*max(row["cost_percent"]-10., 0.) + row["energy_mJ_device_slot"]/baseline["energy_mJ_device_slot"]
                + (100-row["bfsr_percent"])/100)
    chosen = min(eligible, key=score)
    return dict(candidate=chosen, complete=True, metrics=metrics,
                reason="Validation-only multi-metric score; three deployment/training pairs, not final-test selection.")


def confirm_selection(jobs, candidate):
    data = {}
    required = {(1000+l, s) for l in range(3) for s in range(3)}
    for variant in ("shared_exact", candidate):
        rows = {}
        for job in jobs:
            if job["variant"] != variant:
                continue
            summary = completed_summary(job)
            if summary is not None:
                rows[(job["layout_seed"], job["seed"])] = summary
        data[variant] = rows
    if candidate == "shared_exact" or any(set(rows) != required for rows in data.values()):
        return dict(selected="shared_exact", promoted=False,
                    reason="No complete 3-deployment x 3-training-seed confirmation; retain corrected reference.")
    means = {v: {k: float(np.mean([s[k] for s in rows.values()])) for k in
                 ("cost_percent", "energy_mJ_device_slot", "bfsr_percent")}
             for v, rows in data.items()}
    ref, cand = means["shared_exact"], means[candidate]
    deployment_energy_gains = []
    for layout in range(1000, 1003):
        ref_e = np.mean([data["shared_exact"][(layout, s)]["energy_mJ_device_slot"] for s in range(3)])
        new_e = np.mean([data[candidate][(layout, s)]["energy_mJ_device_slot"] for s in range(3)])
        deployment_energy_gains.append(float(1-new_e/ref_e))
    energy_gain = 1-cand["energy_mJ_device_slot"]/ref["energy_mJ_device_slot"]
    stable = all(s["cost_percent"] <= 11.5 for s in data[candidate].values())
    noninferior = cand["cost_percent"] <= min(10., ref["cost_percent"]+.5) and cand["bfsr_percent"] >= ref["bfsr_percent"]-3
    improves = ((energy_gain >= .03 and sum(x > 0 for x in deployment_energy_gains) >= 2)
                or (cand["bfsr_percent"] >= ref["bfsr_percent"]+5 and energy_gain >= -.03))
    passed = bool(stable and noninferior and improves)
    return dict(selected=candidate if passed else "shared_exact", candidate=candidate,
                promoted=passed, metrics=means, energy_gain=energy_gain,
                per_deployment_energy_gains=deployment_energy_gains,
                reason="Predeclared validation guardrails: feasibility, BFSR non-inferiority, energy or BFSR gain, cross-deployment stability."
                if passed else "Candidate failed validation guardrails; continue baseline diagnosis without claiming an improvement.")
