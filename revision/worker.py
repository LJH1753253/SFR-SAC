"""One isolated revision task: training, held-out test, per-slot optimizer run, or latency benchmark."""
from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

from cloud24 import worker as cloud_worker
from revision.source import write_revision_marker
from v2.runtime import read_json, write_json


def main(path):
    spec = read_json(path)
    out = Path(spec["directory"])
    out.mkdir(parents=True, exist_ok=True)
    write_revision_marker(out, spec.get("revision_source_digest"))
    if spec["task"] in {"train", "test"}:
        cloud_worker.main(path)  # unchanged training and held-out evaluation path
        return
    status = dict(status="running", start_time=time.strftime("%Y-%m-%d %H:%M:%S %z"), end_time=None,
                  command=[sys.executable, "-m", "revision.worker", str(path)], pid=os.getpid(), error=None)
    write_json(out / "run_status.json", status)
    try:
        if spec["task"] == "per_slot":
            from revision import per_slot
            per_slot.run(spec, out)
        elif spec["task"] == "latency":
            from revision import latency
            latency.run(spec, out)
        else:
            raise ValueError(f"Unknown task {spec['task']}")
        status["status"] = "done"
    except BaseException:
        status.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        status["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
        write_json(out / "run_status.json", status)


if __name__ == "__main__":
    main(sys.argv[1])
