import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from cloud24 import plan
from v2.checkpoint import source_digest
from v2.runtime import fingerprint, read_json, write_json
from original_ch5.protocol import Catalog, EXPECTED_SOURCE, ROOT, addon_digest
from original_ch5.runner import complete, verify_assets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--require-gpus", type=int, default=1)
    args = p.parse_args()
    if source_digest() != EXPECTED_SOURCE:
        raise RuntimeError("Frozen source mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.require_gpus:
        raise RuntimeError("Required CUDA GPUs are not visible")
    imported_count = verify_assets()
    output = ROOT / "outputs" / f"smoke_{time.time_ns()}"
    catalog = Catalog(output, quick=True)
    checks = []
    for i, (variant, change) in enumerate([
        ("split_exact", None), ("penalty_beta1", None),
        ("split_exact", lambda c: c["algorithms"]["safe"].update(lambda_update_granularity="step")),
        ("split_exact", lambda c: c["algorithms"]["safe"].update(lambda_update_granularity="episode")),
    ]):
        spec = catalog.parent(variant, 8, 0, change)
        directory = Path(spec["directory"])
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "task.json", spec)
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   CUDA_VISIBLE_DEVICES=str(i % torch.cuda.device_count()))
        subprocess.run([sys.executable, "-m", "cloud24.worker", str(directory / "task.json")],
                       cwd=ROOT, env=env, check=True)
        if not complete(spec):
            raise RuntimeError("Tiny training did not complete")
        test = catalog.evaluation(spec)
        target = Path(test["directory"])
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / "task.json", test)
        subprocess.run([sys.executable, "-m", "original_ch5.worker", str(target / "task.json")],
                       cwd=ROOT, env=env, check=True)
        result = read_json(target / "test_summary.json")
        if not complete(test) or result["summary"]["eval_slots"] != 12:
            raise RuntimeError("Tiny test/reload did not complete")
        checks.append(dict(variant=variant, mode=spec["config"]["algorithms"][spec["method"]].get("lambda_update_granularity"),
                           train=str(directory), evaluation=str(target)))
    real = Catalog(output / "real_reload").parent("split_exact", 70, 0)
    if not real["reused"]:
        raise RuntimeError("The shipped N70 split_exact model was not found")
    real_test = Catalog(output / "real_reload").evaluation(real)
    real_test.update(test_episodes=1, directory=str(output / "real_reload_test"))
    real_test["id"] = fingerprint({k: v for k, v in real_test.items() if k != "id"})
    target = Path(real_test["directory"])
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "task.json", real_test)
    subprocess.run([sys.executable, "-m", "original_ch5.worker", str(target / "task.json")], cwd=ROOT, check=True)
    if not complete(real_test) or read_json(target / "test_summary.json")["summary"]["eval_slots"] != 200:
        raise RuntimeError("Imported model reload/evaluation failed")
    receipt = dict(passed=True, source_digest=source_digest(), addon_digest=addon_digest(),
                   imported_models_checked=imported_count, tiny_training_and_reload_checks=checks,
                   real_model_reload=str(target), python=sys.executable, torch=torch.__version__,
                   cuda=torch.version.cuda, gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                   note="Smoke checks only; no formal experiment training started")
    write_json(output / "smoke_receipt.json", receipt)
    print(f"SMOKE PASSED\nReceipt: {output / 'smoke_receipt.json'}", flush=True)


if __name__ == "__main__":
    main()
