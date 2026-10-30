"""Run one SFR-SAC or baseline experiment from a JSON configuration."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v2.plan import job_hash
from v2.runtime import HEURISTICS, LEARNERS, run_job, write_json


METHOD_ALIASES = {
    "sfr_sac": "safe",
    "safe": "safe",
    "penalty_sac": "penalty_sac",
    "td3_lag": "td3_lag",
    "vanilla": "vanilla",
    "lyapunov": "lyapunov",
    "const_ratio": "const_ratio",
    "local_first": "local_first",
    "mb_greedy": "mb_greedy",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.json"))
    parser.add_argument("--method", choices=sorted(METHOD_ALIASES), default="sfr_sac")
    parser.add_argument("--N", type=int, default=70)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout-seed", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("runs/single"))
    args = parser.parse_args()

    if args.N < 1 or args.seed < 0:
        parser.error("N must be positive and seed must be nonnegative")
    cfg = copy.deepcopy(json.loads(args.config.read_text(encoding="utf-8")))
    method = METHOD_ALIASES[args.method]
    if args.method in {"sfr_sac", "safe"}:
        cfg["algorithms"]["safe"]["actor_architecture"] = "split_service_flow"
    layout_seed = args.layout_seed
    if layout_seed is None:
        layout_seed = int(cfg.get("layout_seed_base", 1000)) + args.seed
    cfg.update(
        method=method,
        N=args.N,
        seed=args.seed,
        layout_seed=layout_seed,
        evaluation_seed_base=99000,
        validation_seed_base=88000,
        evaluation_rng_seed=900000 + 100 * args.N + args.seed,
    )
    cfg["execution"] = {
        "device": args.device,
        "require_cuda": args.device.startswith("cuda"),
        "tf32": False,
    }
    if args.episodes is not None:
        cfg["training"]["episodes"] = args.episodes
    if args.eval_episodes is not None:
        cfg["training"]["eval_episodes"] = args.eval_episodes
    if args.horizon is not None:
        cfg["environment"]["max_ep_steps"] = args.horizon

    if method in LEARNERS:
        kind = "learner"
    elif method == "lyapunov":
        kind = "lyapunov"
    elif method in HEURISTICS:
        kind = "heuristic"
    else:
        raise ValueError(f"Unsupported method: {method}")

    args.output.mkdir(parents=True, exist_ok=True)
    job = {
        "kind": kind,
        "method": method,
        "variant": args.method,
        "experiment": "standalone",
        "N": args.N,
        "seed": args.seed,
        "config": cfg,
        "config_hash": job_hash(cfg, method),
        "directory": str(args.output.resolve()),
        "resource": "gpu" if args.device.startswith("cuda") else "cpu",
    }
    write_json(args.output / "job.json", job)
    run_job(args.output / "job.json")


if __name__ == "__main__":
    main()
