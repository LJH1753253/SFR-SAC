import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

from cloud24 import plan
from cloud24.suite import Suite, memory_available_gib
from v2.checkpoint import source_digest
from v2.runtime import fingerprint, read_json, write_json

from original_ch5 import report
from original_ch5.protocol import (BLOCKED_COST, Catalog, EXPECTED_SOURCE, ROOT, STAGES,
                                   addon_digest, definition)


def complete(spec):
    filename = "final_eval_summary.json" if spec["task"] == "train" else "test_summary.json"
    path = Path(spec["directory"]) / filename
    if not path.exists():
        return False
    data = read_json(path)
    if data.get("source_digest") != EXPECTED_SOURCE:
        raise RuntimeError(f"Result source mismatch: {path}")
    if spec["task"] == "train":
        if data.get("config_hash") != spec["config_hash"]:
            raise RuntimeError(f"Result configuration mismatch: {path}")
        if spec["kind"] == "learner" and not (path.parent / "model.pt").exists():
            raise RuntimeError(f"Completed run has no model: {path.parent}")
    elif data.get("id") != spec["id"] or data.get("addon_digest") != addon_digest():
        raise RuntimeError(f"Result test identity mismatch: {path}")
    return True


def verify_assets():
    path = ROOT / "reuse" / "index.json"
    if not path.exists():
        # The public release does not ship the private cloud reuse cache.
        # Missing cache entries are treated as fresh runs rather than as an error.
        return 0
    index = read_json(path)
    if index["source_digest"] != EXPECTED_SOURCE:
        raise RuntimeError("Reused models have incompatible source")
    for value in index["models"].values():
        directory = ROOT / value["directory"]
        for name, expected in value["files"].items():
            source = directory / name
            if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Missing or changed reused asset: {source}")
    return len(index["models"])


class OriginalSuite(Suite):
    def __init__(self, args):
        if source_digest() != EXPECTED_SOURCE:
            raise RuntimeError("Frozen core has changed; refusing to mix experiments")
        super().__init__(args)
        expected = fingerprint(definition(args.quick))
        if self.state.get("protocol_hash", expected) != expected:
            raise RuntimeError("Increment protocol changed; choose a separate output directory")
        self.state.update(protocol_hash=expected, addon_digest=addon_digest(), selected_policy="split_exact")
        duration = self.state["deadline_unix"] - self.state["window_start"]
        self.training_end = self.state["deadline_unix"] - min(3600., duration*.1)
        self.evaluation_end = self.state["deadline_unix"] - min(600., duration*.03)
        self.catalog = Catalog(self.root, args.quick).build()

    def register(self, tasks):
        for spec in tasks:
            directory = Path(spec["directory"])
            if not directory.is_relative_to(self.root):
                raise RuntimeError("Refusing to write to a reused model directory")
            self.specs[spec["id"]] = spec
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "task.json"
            if path.exists() and read_json(path)["id"] != spec["id"]:
                raise RuntimeError(f"Task directory collision: {path}")
            write_json(path, spec)
            if not (directory / "run_status.json").exists():
                write_json(directory / "run_status.json", dict(status="pending", start_time=None,
                           end_time=None, error=None, task_id=spec["id"]))

    def launch(self, spec, gpu):
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES=gpu or "",
                   MPLCONFIGDIR=str(self.root / ".matplotlib"))
        directory = Path(spec["directory"])
        stream = (directory / "worker.log").open("a", encoding="utf-8")
        module = "cloud24.worker" if spec["task"] == "train" else "original_ch5.worker"
        process = subprocess.Popen([sys.executable, "-u", "-m", module, str(directory / "task.json")],
                                   cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        self.active[spec["id"]] = dict(process=process, log=stream, spec=spec, gpu=gpu, start=time.time())
        print(f"START pid={process.pid} gpu={gpu or 'cpu'} {spec['task']} {spec['variant']} "
              f"N={spec['N']} layout={spec['layout_seed']} seed={spec['seed']}", flush=True)

    def poll(self):
        for identity, item in list(self.active.items()):
            code = item["process"].poll()
            if code is None:
                continue
            item["log"].close()
            del self.active[identity]
            spec = item["spec"]
            if complete(spec):
                print(f"DONE {spec['task']} {spec['variant']} N={spec['N']} seed={spec['seed']}", flush=True)
            elif item.get("stop_sent"):
                path = Path(spec["directory"]) / "run_status.json"
                status = read_json(path) if path.exists() else {}
                status.update(status="interrupted", end_time=time.strftime("%Y-%m-%d %H:%M:%S %z"),
                              error="Training time limit reached; resume from the episode checkpoint")
                write_json(path, status)
            elif not self.draining:
                path = Path(spec["directory"]) / "run_status.json"
                status = read_json(path) if path.exists() else {}
                status.update(status="failed", error=status.get("error") or f"Exit {code} without a complete result",
                              end_time=time.strftime("%Y-%m-%d %H:%M:%S %z"))
                write_json(path, status)
                self.failures[identity] = dict(exit_code=code, directory=spec["directory"], detail=status)
                print(f"FAILED {spec['variant']} exit={code}: {spec['directory']}", flush=True)

    def run_original_phase(self, phase, tasks):
        self.register(tasks)
        write_json(self.root / phase / "manifest.json", dict(
            phase=phase, source_digest=source_digest(), addon_digest=addon_digest(), jobs=tasks,
            cases=[c for c in self.catalog.cases if c["experiment"] == phase]))
        self.status(phase=phase, status="running", planned_tasks=len(tasks))
        resource_wait_started = None
        while not self.stop and time.time() < self.evaluation_end:
            self.poll()
            pending = [s for s in tasks if s["id"] not in self.active and s["id"] not in self.failures and not complete(s)]
            if not pending and not self.active:
                break
            if time.time() >= self.training_end:
                for item in list(self.active.values()):
                    if item["spec"]["task"] == "train" and not item.get("stop_sent"):
                        item["process"].terminate()
                        item["stop_sent"] = time.time()
                for item in list(self.active.values()):
                    if item.get("stop_sent") and time.time()-item["stop_sent"] > 180:
                        item["process"].kill()
                pending = [s for s in pending if s["task"] != "train"]
                if not pending and not self.active:
                    break
            counts = {g: sum(i["gpu"] == g for i in self.active.values()) for g in self.gpus}
            gpu_running = sum(i["spec"]["resource"] == "gpu" for i in self.active.values())
            cpu_running = len(self.active)-gpu_running
            blocked_resource = memory_available_gib() < self.args.min_free_ram_gib or shutil.disk_usage(self.root).free < 5*1024**3
            if blocked_resource and not self.active:
                resource_wait_started = resource_wait_started or time.time()
                if time.time()-resource_wait_started > 60:
                    self.state["resource_error"] = "Insufficient free RAM or less than 5 GiB disk; resume after freeing resources"
                    break
            else:
                resource_wait_started = None
            for spec in pending:
                if blocked_resource or not self.ready(spec):
                    continue
                if spec["resource"] == "cpu":
                    if cpu_running >= self.args.cpu_workers:
                        continue
                    self.launch(spec, None)
                    cpu_running += 1
                elif gpu_running < self.workers:
                    gpu = min(counts, key=counts.get)
                    self.launch(spec, gpu)
                    counts[gpu] += 1
                    gpu_running += 1
            if not self.active and pending and all(not self.ready(s) for s in pending):
                break
            if time.time()-self.last_report >= self.args.report_seconds:
                self.status(completed_tasks=sum(complete(s) for s in tasks))
                report.generate(self.root)
                self.last_report = time.time()
            time.sleep(.05 if self.args.quick else 2)
        if self.active:
            self.drain()
        self.status(completed_tasks=sum(complete(s) for s in tasks))
        report.generate(self.root)

    def execute(self):
        verify_assets()
        manifest = self.catalog.manifest()
        write_json(self.root / "manifest.json", manifest)
        write_json(self.root / "protocol.json", definition(self.args.quick))
        write_json(self.root / "blocked_experiments.json", manifest["protocol"]["blocked_experiments"])
        phases = STAGES if self.args.stage == "all" else (self.args.stage,)
        if self.args.stage == "cost_target":
            print(BLOCKED_COST, flush=True)
            self.status(status="blocked_cost_target_configuration")
            return
        for phase in STAGES:
            cases = [c for c in self.catalog.cases if c["experiment"] == phase]
            ids = {c["parent_id"] for c in cases}
            new = sum(not self.catalog.parents[i]["reused"] for i in ids)
            print(f"PLAN {phase}: {len(cases)} case rows; {new} non-imported training identities; "
                  f"{len(ids)-new} imported models; shared defaults are deduplicated", flush=True)
        if self.args.dry_run:
            self.status(status="dry_run")
            report.generate(self.root)
            return
        import fcntl
        self.lock = (self.root / "suite.lock").open("a+")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.owns_lock = True
        (self.root / "suite.pid").write_text(str(os.getpid())+"\n")
        if self.args.stage in {"report", "export"}:
            if self.args.stage == "export":
                self.export()
            else:
                report.generate(self.root, final=True)
            return
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing CPU fallback")
        for gpu in self.gpus:
            torch.cuda.get_device_name(int(gpu))
        write_json(self.root / "hardware.json", dict(python=sys.executable, torch=torch.__version__,
                   cuda=torch.version.cuda, gpus=[torch.cuda.get_device_name(int(g)) for g in self.gpus],
                   cpu_count=os.cpu_count()))
        for phase in phases:
            if self.stop or time.time() >= self.evaluation_end:
                break
            self.run_original_phase(phase, self.catalog.tasks(phase))
        requested = {s["id"]: s for phase in phases for s in self.catalog.tasks(phase)}
        done_count = sum(complete(s) for s in requested.values())
        self.status(status="interrupted" if self.stop else "requested_stages_finished" if done_count == len(requested)
                    else "budget_finished" if time.time() >= self.training_end else "incomplete_with_failures_or_resources",
                    requested_tasks=len(requested), requested_completed=done_count)
        self.export()

    def export(self):
        report.generate(self.root, final=True)
        write_json(self.root / "runtime_manifest.json", dict(
            source_digest=source_digest(), addon_digest=addon_digest(), args=vars(self.args),
            imported_models_read_only=True, no_policy_selection=True, python=sys.version))
        archive = self.root.parent / (self.root.name+"_results.tar.gz")
        temporary = archive.with_suffix(".tmp")
        with tarfile.open(temporary, "w:gz", compresslevel=1) as stream:
            for path in self.root.rglob("*"):
                if path.is_file() and path.name not in {"checkpoint.pt", "suite.lock"} and ".matplotlib" not in path.parts and "evaluation_work" not in path.parts:
                    stream.add(path, arcname=str(Path(self.root.name) / path.relative_to(self.root)))
            for spec in self.catalog.parents.values():
                if not spec["reused"]:
                    continue
                parent = Path(spec["directory"])
                for name in ("job.json", "final_eval_summary.json", "model.pt", "environment.json", "training.csv"):
                    path = parent / name
                    if path.exists():
                        stream.add(path, arcname=str(Path(self.root.name) / "reused" / spec["id"] / name))
        temporary.replace(archive)
        print(f"EXPORT {archive}", flush=True)


def arguments():
    p = argparse.ArgumentParser(description="Complete the original experiment grids with the locked split_exact policy")
    p.add_argument("--output", default="outputs/original_ch5")
    p.add_argument("--hours", type=float, default=24.)
    p.add_argument("--extend-hours", type=float, default=0.)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--cpu-workers", type=int, default=2)
    p.add_argument("--min-free-ram-gib", type=float, default=12.)
    p.add_argument("--report-seconds", type=float, default=300.)
    p.add_argument("--stage", choices=("all", *STAGES, "cost_target", "report", "export"), default="all")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if a.hours <= 0 or a.extend_hours < 0 or a.workers < 1 or a.cpu_workers < 1 or a.report_seconds <= 0:
        p.error("Invalid worker count or time limit")
    if a.quick and a.output == "outputs/original_ch5":
        a.output = "outputs/original_ch5_quick"
    return a


def main():
    suite = OriginalSuite(arguments())
    try:
        suite.execute()
    except BaseException:
        if suite.owns_lock:
            import traceback
            suite.drain()
            suite.status(status="failed", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
