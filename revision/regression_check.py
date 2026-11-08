"""Check that the revision branch leaves every main-grid method unchanged.

    git worktree add ../sfr-sac-main main
    python -m revision.regression_check ../sfr-sac-main .

Trains each main-grid arm briefly (tiny CPU configuration) in both checkouts, in separate processes,
and compares a hash of the trained weights and the evaluation summary. Identical output means that
the frozen results of `original_ch5` remain valid next to the revision study.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT = r'''
import hashlib, os, sys, tempfile
from pathlib import Path
sys.path.insert(0, os.getcwd())
import torch
from cloud24 import plan
from v2.runtime import run_heuristic, run_learner, run_lyapunov
for variant in ["split_exact", "shared_exact", "td3", "penalty_beta5", "vanilla", "mlp_critics",
                "sac_lagrangian", "mb_greedy", "dpp_rule"]:
    cfg = plan.configure(variant, n=5, layout=1000, train_seed=0, selected="split_exact", small=True)
    cfg["execution"] = dict(device="cpu", require_cuda=False, tf32=False)
    method, out = cfg["method"], Path(tempfile.mkdtemp())
    if method in ("safe", "td3_lag", "penalty_sac", "vanilla"):
        summary = run_learner(cfg, method, 5, 0, out, device="cpu")
        weights = torch.load(out / "model.pt", weights_only=True)["modules"]
        digest = hashlib.sha256()
        for module in sorted(weights):
            for key, value in sorted(weights[module].items()):
                digest.update(f"{module}.{key}".encode())
                digest.update(value.cpu().numpy().tobytes())
        tag = digest.hexdigest()[:16]
    elif method == "lyapunov":
        summary, tag = run_lyapunov(cfg, 5, 0, out), "-"
    else:
        summary, tag = run_heuristic(cfg, method, 5, 0, out), "-"
    print("RESULT", variant, tag, repr(summary["cost_percent"]), repr(summary["energy_mJ_device_slot"]), flush=True)
'''


def run(checkout):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-c", SCRIPT], cwd=checkout, capture_output=True, text=True, env=env)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])
    return [line for line in result.stdout.splitlines() if line.startswith("RESULT")]


def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) != 2:
        raise SystemExit("usage: python -m revision.regression_check <reference checkout> <revision checkout>")
    reference, revision = (run(Path(p).resolve()) for p in argv)
    for a, b in zip(reference, revision):
        print(("SAME  " if a == b else "DIFF  ") + b[7:])
    if reference != revision:
        raise SystemExit("Main-grid methods changed")
    print("All main-grid methods are bitwise identical in the two checkouts.")


if __name__ == "__main__":
    main()
