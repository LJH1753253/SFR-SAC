"""Explicit, source-checked checkpoint forks for declared objective interventions."""

import copy
from pathlib import Path

import torch

from v2.checkpoint import atomic_save, source_digest
from v2.runtime import ACTOR_DIAGNOSTIC_FIELDS, AP_DIAGNOSTIC_FIELDS, read_json, write_json

ALLOWED_ALGORITHM_FIELDS = {"penalty_mode", "training_cost_limit", "dual_cost_source", "dual_risk_mode",
                            "excess_budget", "constraint_diagnostics", "actor_grad_clip",
                            "actor_update_interval", "actor_stability_diagnostics", "freeze_alpha",
                            "replay_recent_fraction", "replay_recent_window", "ap_entropy_mode",
                            "freeze_ap_policy", "ap_branch_diagnostics"}
IDENTICAL_FILES = ("agents/networks.py", "agents/action_projection.py", "agents/structured_cost_critic.py",
                   "agents/structured_energy_critic.py", "v2/env.py", "v2/replay.py", "offloading/runtime.py")


def fork_checkpoint(cfg, destination, identity, device):
    declaration = cfg["intervention"]
    parent = Path(declaration["parent_directory"])
    project = Path(declaration["parent_project"])
    current = Path(__file__).resolve().parents[1]
    job = read_json(parent / "job.json")
    previous = job["config"]
    completed = read_json(parent / "final_eval_summary.json")
    if completed["source_digest"] != source_digest(project) or completed["config_hash"] != job["config_hash"]:
        raise ValueError("Ancestor source/result identity mismatch")
    for name in IDENTICAL_FILES:
        if (project / name).read_bytes() != (current / name).read_bytes():
            raise ValueError(f"Intervention changed a protected model or environment file: {name}")
    comparable = copy.deepcopy(cfg)
    comparable.pop("intervention")
    comparable["training"]["episodes"] = previous["training"]["episodes"]
    for field in ("evaluation_seed_base", "validation_seed_base"):
        if field in previous:
            comparable[field] = previous[field]
        else:
            comparable.pop(field, None)
    for field in ALLOWED_ALGORITHM_FIELDS:
        if field in previous["algorithms"]["safe"]:
            comparable["algorithms"]["safe"][field] = previous["algorithms"]["safe"][field]
        else:
            comparable["algorithms"]["safe"].pop(field, None)
    if comparable != previous or cfg["training"]["episodes"] <= previous["training"]["episodes"]:
        raise ValueError("Checkpoint fork contains undeclared changes")
    saved = torch.load(parent / "checkpoint.pt", map_location=device, weights_only=False)
    if saved.get("complete") or saved["identity"] != job["config_hash"]:
        raise ValueError("A full matching ancestor checkpoint is required")
    if saved["episode"] != declaration["parent_episode"]:
        raise ValueError("Ancestor checkpoint is not the declared episode")
    saved["identity"] = identity
    if cfg["algorithms"]["safe"].get("freeze_ap_policy", False):
        original = saved["agent"]["modules"]["actor"]
        if any(key.startswith("policy.") or key.startswith("ap_policy.") for key in original):
            raise ValueError("AP freeze diagnostic expects an unwrapped ancestor actor")
        saved["agent"]["modules"]["actor"] = {
            **{"policy."+key:value for key,value in original.items()},
            **{"ap_policy."+key:value.detach().clone() for key,value in original.items()},
        }
    for row in saved["history"]:
        for key in ("dual_statistic", "dual_budget", "dual_replay_cost_mean", "cost_pi_mean"):
            row[key] = None
        if cfg["algorithms"]["safe"].get("actor_stability_diagnostics", False):
            for key in ACTOR_DIAGNOSTIC_FIELDS:
                row[key] = None
        if cfg["algorithms"]["safe"].get("ap_branch_diagnostics", False):
            for key in AP_DIAGNOSTIC_FIELDS:
                row[key] = None
    atomic_save(destination, saved)
    write_json(Path(destination).with_name("intervention_provenance.json"), dict(
        parent=str(parent), parent_episode=saved["episode"], ancestor_source_digest=completed["source_digest"],
        ancestor_identity=job["config_hash"], new_identity=identity, protected_files=list(IDENTICAL_FILES),
        inherited=["models", "target_models", "optimizers", "replay", "RNG", "lambda", "alpha"],
        ap_snapshot_from_parent=cfg["algorithms"]["safe"].get("freeze_ap_policy", False),
        history_diagnostics_before_intervention="not measured", declaration=declaration))
