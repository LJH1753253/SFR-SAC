import sys
import time
from pathlib import Path

from cloud24 import worker
from v2.runtime import read_json, write_json

from original_ch5.protocol import addon_digest


heldout = worker.heldout


def evaluate(spec, out):
    if spec["addon_digest"] != addon_digest():
        raise RuntimeError("Increment changed after the experiment was planned")
    staging = Path(out) / "evaluation_work"
    staging.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    heldout(spec, staging)
    result = read_json(staging / "test_summary.json")
    result.update(split=spec["split"], addon_digest=addon_digest(),
                  evaluation_wall_seconds=time.monotonic()-started,
                  note="Fixed-policy test; no candidate/default selection")
    for name in ("eval_episodes.csv", "eval_slots.csv", "per_device.csv"):
        (staging / name).replace(Path(out) / name)
    write_json(Path(out) / "test_summary.json", result)


if __name__ == "__main__":
    worker.heldout = evaluate
    worker.main(sys.argv[1])
