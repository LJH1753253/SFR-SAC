"""CUDA checks and short real training; no local datasets or pretrained models required."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from cloud24.plan import CANDIDATES, test_job, training_job
from cloud24.report import generate
from v2.checkpoint import source_digest
from v2.runtime import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--output", default="outputs/smoke")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    variants = (*CANDIDATES, "sac_lagrangian", "penalty_beta5", "vanilla", "td3")
    gpus = args.gpus.split(",")
    results = []
    for index, variant in enumerate(variants):
        spec = training_job(root, variant, n=70, selected="split_bounded", small=True)
        path = Path(spec["directory"])
        path.mkdir(parents=True, exist_ok=True)
        write_json(path/"task.json", spec)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpus[index % len(gpus)], OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        print(f"SMOKE {variant} GPU={env['CUDA_VISIBLE_DEVICES']}", flush=True)
        subprocess.run([sys.executable, "-m", "cloud24.worker", str(path/"task.json")], env=env, check=True)
        if not (path/"final_eval_summary.json").exists():
            raise RuntimeError(f"Smoke did not finish: {variant}")
        test = test_job(root, spec, variant, small=True)
        test_path = Path(test["directory"])
        test_path.mkdir(parents=True, exist_ok=True)
        write_json(test_path/"task.json", test)
        subprocess.run([sys.executable, "-m", "cloud24.worker", str(test_path/"task.json")], env=env, check=True)
        if not (test_path/"test_summary.json").exists():
            raise RuntimeError(f"Held-out pipeline check failed: {variant}")
        results.append(dict(variant=variant, assigned_gpu=env["CUDA_VISIBLE_DEVICES"]))
    write_json(root/"verified.json", dict(source_digest=source_digest(), timestamp=time.time(),
               variants=results, N=70, note="Short pipeline check only; not a performance experiment."))
    generate(root, final=True)
    print(f"SMOKE PASSED: {root/'verified.json'}", flush=True)


if __name__ == "__main__":
    main()
