"""Evaluate a shipped SFR-SAC checkpoint deterministically."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from v2.runtime import evaluate_agent, load_agent, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/sfr_sac_N70_seed0/model.pt"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("runs/example_evaluation"))
    parser.add_argument("--allow-source-mismatch", action="store_true",
                        help="load a checkpoint saved by a different source version (the configuration must match)")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")

    job_path = args.model.parent / "job.json"
    if not job_path.exists():
        raise FileNotFoundError(f"Expected checkpoint metadata: {job_path}")
    job = json.loads(job_path.read_text(encoding="utf-8"))
    cfg = copy.deepcopy(job["config"])
    agent, method, n, seed = load_agent(args.model, cfg, device=args.device,
                                        check_source=not args.allow_source_mismatch)
    cfg["training"]["eval_episodes"] = args.episodes
    args.output.mkdir(parents=True, exist_ok=True)
    summary = evaluate_agent(cfg, method, agent, n, seed, args.episodes, details_dir=args.output)
    write_json(args.output / "final_eval_summary.json", {
        "model": str(args.model), "method": method, "N": n, "seed": seed, "summary": summary,
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
