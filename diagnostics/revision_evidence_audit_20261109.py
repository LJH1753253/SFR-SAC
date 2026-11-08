"""Read-only consistency and mechanism diagnostics for the IoT-J revision evidence."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
OLD = Path(r"D:/PythonCode/revision_original_ch5_20261109/original_ch5")
FAIR = Path(r"D:/PythonCode/SFR-SAC-revision-preflight/outputs/fair_critics_20261107_v3")
REMAINING = Path(r"D:/PythonCode/revision_remaining_results_20261108/outputs/revision_remaining_20261107")
OUT = REPO / "diagnostics" / "revision_evidence_audit_20261109"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from v2.runtime import load_agent, make_env, seed_all


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def old_main_rows():
    rows = []
    with (OLD / "main" / "raw.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["variant"] == "split_exact" and int(row["N"]) == 70:
                rows.append({
                    "seed": int(row["seed"]),
                    "layout_seed": int(row["layout_seed"]),
                    "cost_percent": float(row["cost_percent"]),
                    "energy_mJ_device_slot": float(row["energy_mJ_device_slot"]),
                    "bfsr_percent": float(row["bfsr_percent"]),
                    "avg_offload_ratio": float(row["avg_offload_ratio"]),
                    "snr_db_active_mean": float(row["snr_db_active_mean"]),
                    "best_ap_percent": float(row["best_ap_percent"]),
                    "ap_load_cv": float(row["ap_load_cv"]),
                    "test_id": row["test_id"],
                    "config_hash": row["parent_id"],
                    "source_digest": read_json(OLD / "test" / row["test_id"] / "test_summary.json")["source_digest"],
                })
    return sorted(rows, key=lambda row: row["seed"])


def fair_sfr_rows():
    rows = []
    for path in (FAIR / "test").glob("*/test_summary.json"):
        payload = read_json(path)
        spec = payload["spec"]
        if spec.get("variant") != "sfr":
            continue
        s = payload["summary"]
        rows.append({
            "seed": int(spec["seed"]),
            "layout_seed": int(spec["layout_seed"]),
            "cost_percent": float(s["cost_percent"]),
            "energy_mJ_device_slot": float(s["energy_mJ_device_slot"]),
            "bfsr_percent": float(s["bfsr_percent"]),
            "avg_offload_ratio": float(s["avg_offload_ratio"]),
            "snr_db_active_mean": float(s["snr_db_active_mean"]),
            "best_ap_percent": float(s["best_ap_percent"]),
            "ap_load_cv": float(s["ap_load_cv"]),
            "test_id": payload["id"],
            "config_hash": spec["parent_id"],
            "source_digest": payload["source_digest"],
        })
    return sorted(rows, key=lambda row: row["seed"])


def compare_configs():
    old_manifest = read_json(OLD / "manifest.json")
    old_eval = next(
        e for e in old_manifest["evaluations"]
        if e["N"] == 70 and e["seed"] == 0 and e["variant"] == "split_exact"
        and e["config_hash"] == "3efb5379d8a8e913"
    )
    new_job = read_json(FAIR / "runs" / "N70" / "layout1000" / "train0" / "24b8bc3f4b1adf2f" / "job.json")
    old_cfg = old_eval["config"]
    new_cfg = new_job["config"]
    old_cfg = json.loads(json.dumps(old_cfg))
    new_cfg = json.loads(json.dumps(new_cfg))
    new_cfg.pop("revision_variant", None)
    old_cfg.pop("revision_variant", None)

    differences = []

    def walk(left, right, path="config"):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                if key not in left or key not in right:
                    differences.append({"path": f"{path}.{key}", "old": left.get(key), "new": right.get(key)})
                else:
                    walk(left[key], right[key], f"{path}.{key}")
        elif left != right:
            differences.append({"path": path, "old": left, "new": right})

    walk(old_cfg, new_cfg)
    return {
        "configs_equal_after_revision_variant_removal": old_cfg == new_cfg,
        "config_differences": differences,
        "old_config_hash": old_eval["config_hash"],
        "new_config_hash": new_job["config_hash"],
        "old_source_digest": read_json(OLD / "test" / old_eval["directory"].rstrip("/").split("/")[-1] / "test_summary.json")["source_digest"],
        "new_source_digest": new_job.get("source_digest", read_json(FAIR / "runs" / "N70" / "layout1000" / "train0" / "24b8bc3f4b1adf2f" / "final_eval_summary.json")["source_digest"]),
        "old_hardware": read_json(OLD / "hardware.json"),
        "new_hardware": read_json(FAIR / "runs" / "N70" / "layout1000" / "train0" / "24b8bc3f4b1adf2f" / "hardware.json"),
        "old_training_episodes": old_cfg["training"]["episodes"],
        "new_training_episodes": new_cfg["training"]["episodes"],
        "old_eval_stream": old_eval["test_rng_seed"],
        "new_eval_stream": 2907000,
    }


def run_ap_diagnostic(variant: str, seed: int = 0, episodes: int = 5, steps: int = 200):
    job_path = None
    for candidate in REMAINING.glob(f"runs/N70/layout100{seed}/train{seed}/*/job.json"):
        candidate_job = read_json(candidate)
        if candidate_job["config"].get("revision_variant") == variant:
            job_path = candidate
            break
    if job_path is None:
        raise RuntimeError(f"No {variant} job found for seed {seed}")
    job = read_json(job_path)
    model_path = job_path.parent / "model.pt"
    agent, _, _, _ = load_agent(model_path, job["config"], device="cpu")
    env = make_env(job["config"], 70, seed)
    seed_all(2907000 + seed)
    selected = Counter()
    active = Counter()
    tie_rows = 0
    logit_rows = 0
    total_slots = 0
    ap_cvs = []
    costs = []
    for _ in range(episodes):
        state = env.reset()
        for _ in range(steps):
            state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                _, _, logits = agent.actor(state_t)
            logits = logits.squeeze(0).cpu().numpy()
            tie_rows += int(np.mean(np.max(logits, axis=1) - np.min(logits, axis=1) < 1e-7) > 0.99)
            logit_rows += 1
            action = agent.select_action(state, deterministic=True)
            state, _, _, info = env.step(action)
            ap = np.asarray(info["ap_index_vec"], dtype=int)
            counts = np.asarray(info["ap_counts"], dtype=int)
            for value in ap:
                selected[int(value)] += 1
            for index, value in enumerate(counts):
                active[index] += int(value)
            if counts.sum() > 0:
                ap_cvs.append(float(np.std(counts) / max(np.mean(counts), 1e-9)))
            costs.append(float(info["service_failure_ratio"]))
            total_slots += 1
            if getattr(env, "ep_step_count", 0) >= env.max_ep_steps:
                break
    return {
        "variant": variant,
        "seed": seed,
        "episodes": episodes,
        "steps_requested": steps,
        "slots_observed": total_slots,
        "selected_ap_counts_all_devices": dict(sorted(selected.items())),
        "active_ap_counts": dict(sorted(active.items())),
        "mean_active_ap_load_cv": float(np.mean(ap_cvs)),
        "max_active_ap_load_cv": float(np.max(ap_cvs)),
        "mean_cost_percent": float(100 * np.mean(costs)),
        "tie_heavy_slot_fraction": float(tie_rows / max(logit_rows, 1)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old = old_main_rows()
    new = fair_sfr_rows()
    old_by_seed, new_by_seed = {r["seed"]: r for r in old}, {r["seed"]: r for r in new}
    paired = []
    for seed in sorted(set(old_by_seed) & set(new_by_seed)):
        left, right = old_by_seed[seed], new_by_seed[seed]
        paired.append({
            "seed": seed,
            "layout_seed": left["layout_seed"],
            "old_cost_percent": left["cost_percent"],
            "new_cost_percent": right["cost_percent"],
            "old_energy_mJ_device_slot": left["energy_mJ_device_slot"],
            "new_energy_mJ_device_slot": right["energy_mJ_device_slot"],
            "old_bfsr_percent": left["bfsr_percent"],
            "new_bfsr_percent": right["bfsr_percent"],
            "delta_cost_percent": right["cost_percent"] - left["cost_percent"],
            "delta_energy_mJ_device_slot": right["energy_mJ_device_slot"] - left["energy_mJ_device_slot"],
            "delta_bfsr_percent": right["bfsr_percent"] - left["bfsr_percent"],
            "old_snr_db": left["snr_db_active_mean"],
            "new_snr_db": right["snr_db_active_mean"],
            "old_avg_offload": left["avg_offload_ratio"],
            "new_avg_offload": right["avg_offload_ratio"],
            "old_ap_load_cv": left["ap_load_cv"],
            "new_ap_load_cv": right["ap_load_cv"],
        })
    (OUT / "original_vs_fair_sfr.csv").write_text(
        "".join([",".join(map(str, row.values())) + "\n" for row in paired]), encoding="utf-8"
    )
    diagnostics = {
        "paired_results": paired,
        "config_and_hardware": compare_configs(),
        "ap_independent_seed0": run_ap_diagnostic("sfr_apind", seed=0),
        "ap_dependent_seed0": run_ap_diagnostic("sfr", seed=0),
        "max_gain_seed0": run_ap_diagnostic("sfr_maxgain", seed=0),
        "source_code_note": "Original source digest e942... is not present in the current checkout; exact source-level equivalence cannot be reconstructed from the results archive alone.",
        "formal_outputs_modified": False,
    }
    (OUT / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
