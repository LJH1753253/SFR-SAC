"""Resumable scheduler for the revision study (GPU training, CPU tests/optimizers/benchmarks).

    python -m revision.runner --output outputs/revision --resume --gpus 0,1 --workers 8 --cpu-workers 4
    python -m revision.runner --output outputs/revision --dry-run
    python -m revision.runner --output outputs/revision --stage report

The output directory records the source digest when it is created and refuses to continue with
different code, so results from different code versions are never mixed.
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from revision import report
from revision.protocol import ROOT, STAGES, Catalog, definition
from revision.source import revision_source_digest, write_revision_marker
from v2.checkpoint import source_digest
from v2.runtime import fingerprint, read_json, write_json

# Relative training cost, used only for the time estimate (SFR-SAC at N=70 = 1).
COST = {"safe": 1.0, "td3_lag": .6, "penalty_sac": .8, "vanilla": .8, "cpo": .3, "mb_greedy": .03,
        "lyapunov": .03, "const_ratio": .03, "local_first": .01}


def memory_available_gib():
    try:
        values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        return int(values["MemAvailable"].split()[0]) / 1024 ** 2
    except (OSError, KeyError, ValueError):
        return 64.


def complete(spec):
    marker = Path(spec["directory"]) / "revision_source.json"
    expected_revision = spec.get("revision_source_digest")
    if expected_revision is None:
        expected_revision = revision_source_digest()
    if not marker.exists() or read_json(marker).get("revision_source_digest") != expected_revision:
        return False
    if spec["task"] == "train":
        path = Path(spec["directory"]) / "final_eval_summary.json"
        if not path.exists():
            return False
        data = read_json(path)
        if data.get("config_hash") != spec["config_hash"] or data.get("source_digest") != source_digest():
            raise RuntimeError(f"Result identity mismatch: {path}")
        return spec["kind"] != "learner" or (path.parent / "model.pt").exists()
    path = Path(spec["directory"]) / "test_summary.json"
    if not path.exists():
        return False
    data = read_json(path)
    if data.get("id") != spec["id"] or data.get("source_digest") != source_digest():
        raise RuntimeError(f"Result identity mismatch: {path}")
    return True


def ready(spec):
    if spec["task"] == "train":
        return True
    return (Path(spec["parent"]) / "final_eval_summary.json").exists()


class RevisionSuite:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.output).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
        self.active, self.failures = {}, {}
        self.stop = False
        state_path = self.root / "suite_status.json"
        old = read_json(state_path) if state_path.exists() else None
        protocol = fingerprint(definition(args.quick))
        if old and not (args.resume or args.dry_run or args.stage == "report"):
            raise RuntimeError("Output exists; use --resume or a new output directory")
        if old and old["source_digest"] != source_digest():
            raise RuntimeError("Source code changed since this output was created; use a new output directory")
        if old and old.get("revision_source_digest") != revision_source_digest():
            raise RuntimeError("Revision extension code changed since this output was created; use a new output directory")
        if old and old["protocol_hash"] != protocol:
            raise RuntimeError("Protocol changed; use a new output directory")
        self.state = old or dict(created=time.strftime("%Y-%m-%d %H:%M:%S %z"), source_digest=source_digest(),
                                 revision_source_digest=revision_source_digest(), protocol_hash=protocol,
                                 quick=args.quick)
        self.state["deadline_unix"] = None if args.hours is None else time.time() + args.hours * 3600
        self.failures = {} if args.retry_failed else dict(self.state.get("failed", {}))
        self.catalog = Catalog(self.root, args.quick).build()
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._request_stop)

    def _request_stop(self, *_):
        self.stop = True

    def status(self, **extra):
        self.state.update(extra, failed=self.failures, updated=time.strftime("%Y-%m-%d %H:%M:%S %z"),
                          pid=os.getpid(), active=[dict(id=k, variant=v["spec"]["variant"], task=v["spec"]["task"],
                                                        gpu=v["gpu"]) for k, v in self.active.items()])
        write_json(self.root / "suite_status.json", self.state)

    def register(self, tasks):
        for spec in tasks:
            directory = Path(spec["directory"])
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "task.json"
            if path.exists() and read_json(path)["id"] != spec["id"]:
                raise RuntimeError(f"Task directory collision: {path}")
            write_json(path, spec)

    def launch(self, spec, gpu):
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                   PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES=gpu or "", MPLCONFIGDIR=str(self.root / ".matplotlib"))
        directory = Path(spec["directory"])
        stream = (directory / "worker.log").open("a", encoding="utf-8")
        process = subprocess.Popen([sys.executable, "-u", "-m", "revision.worker", str(directory / "task.json")],
                                   cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        self.active[spec["id"]] = dict(process=process, log=stream, spec=spec, gpu=gpu, start=time.time())
        print(f"START {spec['task']} {spec['variant']} N={spec['N']} layout={spec['layout_seed']} "
              f"seed={spec['seed']} gpu={gpu or 'cpu'}", flush=True)

    def poll(self):
        for identity, item in list(self.active.items()):
            code = item["process"].poll()
            if code is None:
                continue
            item["log"].close()
            del self.active[identity]
            spec = item["spec"]
            if complete(spec):
                print(f"DONE {spec['task']} {spec['variant']} N={spec['N']} seed={spec['seed']} "
                      f"({(time.time() - item['start']) / 60:.0f} min)", flush=True)
            elif not item.get("stopping"):
                self.failures[identity] = dict(exit_code=code, directory=spec["directory"], variant=spec["variant"])
                print(f"FAILED {spec['task']} {spec['variant']} exit={code}: {spec['directory']}/worker.log", flush=True)

    def drain(self):
        for item in self.active.values():
            item["stopping"] = True
            item["process"].terminate()
        deadline = time.time() + 180
        while self.active and time.time() < deadline:
            self.poll()
            time.sleep(1)
        for item in self.active.values():
            item["process"].kill()
            item["process"].wait()
            item["log"].close()
        self.active.clear()

    def run_stage(self, stage):
        tasks = self.catalog.tasks(stage)
        self.register(tasks)
        write_json(self.root / stage / "manifest.json", dict(stage=stage, source_digest=source_digest(),
                   tasks=tasks, cases=[c for c in self.catalog.cases if c["experiment"] == stage]))
        need_gpu = any(t["resource"] == "gpu" and not complete(t) for t in tasks)
        if need_gpu and not self.gpus:
            raise RuntimeError(f"Stage {stage} needs GPUs; pass --gpus")
        deadline = self.state["deadline_unix"]
        while not self.stop and (deadline is None or time.time() < deadline):
            self.poll()
            pending = [t for t in tasks if t["id"] not in self.active and t["id"] not in self.failures
                       and not complete(t)]
            if not pending and not self.active:
                break
            gpu_load = {g: sum(i["gpu"] == g for i in self.active.values()) for g in self.gpus}
            gpu_running = sum(i["spec"]["resource"] == "gpu" for i in self.active.values())
            cpu_running = len(self.active) - gpu_running
            low = memory_available_gib() < self.args.min_free_ram_gib or shutil.disk_usage(self.root).free < 5 * 1024 ** 3
            if low and not self.active and not getattr(self, "_warned_low", False):
                print(f"WAIT: less than {self.args.min_free_ram_gib} GiB free RAM or 5 GiB disk; "
                      "lower --min-free-ram-gib or free resources", flush=True)
                self._warned_low = True
            for spec in pending:
                if low or not ready(spec):
                    continue
                if spec["resource"] == "cpu":
                    if cpu_running < self.args.cpu_workers:
                        self.launch(spec, None)
                        cpu_running += 1
                elif gpu_running < self.args.workers:
                    gpu = min(gpu_load, key=gpu_load.get)
                    self.launch(spec, gpu)
                    gpu_load[gpu] += 1
                    gpu_running += 1
            if not self.active and pending and not any(ready(s) for s in pending):
                break  # remaining tasks depend on failed parents
            self.status(stage=stage, completed=sum(complete(t) for t in tasks), planned=len(tasks))
            time.sleep(.2 if self.args.quick else 5)
        if self.active:
            print(f"STOP {stage}: terminating running tasks (training resumes from checkpoints)", flush=True)
            self.drain()
        self.status(stage=stage, completed=sum(complete(t) for t in tasks), planned=len(tasks))

    def estimate(self):
        lines, gpu_cost, cpu_tasks = [], 0.0, 0
        for stage in STAGES:
            tasks = [t for t in self.catalog.tasks(stage)]
            todo = [t for t in tasks if not complete(t)]
            lines.append(f"PLAN {stage:13s} {len(tasks):4d} tasks, {len(todo):4d} to run")
        unique = {t["id"]: t for s in STAGES for t in self.catalog.tasks(s) if not complete(t)}
        for t in unique.values():
            if t["task"] == "train" and t["resource"] == "gpu":
                gpu_cost += COST.get(t["method"], 1.0)
            elif t["resource"] == "cpu":
                cpu_tasks += 1
        lines.append(f"PLAN unique: {sum(t['task'] == 'train' for t in unique.values())} trainings "
                     f"(GPU cost {gpu_cost:.0f} SFR-SAC-equivalents), {cpu_tasks} CPU tasks")
        lines.append("PLAN at about 2.5 h per SFR-SAC-equivalent with 8 concurrent GPU runs: "
                     f"about {gpu_cost * 2.5 / max(self.args.workers, 1):.0f} h of GPU time")
        return lines

    def execute(self):
        write_json(self.root / "manifest.json", self.catalog.manifest())
        write_json(self.root / "protocol.json", definition(self.args.quick))
        for line in self.estimate():
            print(line, flush=True)
        if self.args.dry_run:
            return
        if self.args.stage == "report":
            report.generate(self.root)
            return
        stages = STAGES if self.args.stage == "all" else (self.args.stage,)
        for stage in stages:
            if self.stop or (self.state["deadline_unix"] is not None and
                             time.time() >= self.state["deadline_unix"]):
                break
            print(f"STAGE {stage}", flush=True)
            self.run_stage(stage)
            report.generate(self.root)
        self.status(status="stopped" if self.stop else "finished")
        report.generate(self.root)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default="outputs/revision")
    p.add_argument("--stage", choices=("all", *STAGES, "report"), default="all")
    p.add_argument("--gpus", default="0")
    p.add_argument("--workers", type=int, default=8, help="concurrent GPU training runs")
    p.add_argument("--cpu-workers", type=int, default=4)
    p.add_argument("--hours", type=float, default=None,
                   help="optional wall-clock deadline; omitted means no time limit")
    p.add_argument("--min-free-ram-gib", type=float, default=8.)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quick", action="store_true", help="tiny CPU-only rehearsal (meaningless numbers)")
    a = p.parse_args(argv)
    if a.quick and a.output == "outputs/revision":
        a.output = "outputs/revision_quick"
    if a.quick and a.min_free_ram_gib == 8.:
        a.min_free_ram_gib = .5
    return a


def main(argv=None):
    suite = RevisionSuite(arguments(argv))
    try:
        suite.execute()
    except BaseException:
        suite.drain()
        raise


if __name__ == "__main__":
    main()
