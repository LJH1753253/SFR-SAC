"""Multi-GPU scheduling with validation gates, a fixed deadline, and resumable jobs."""

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

from cloud24 import plan
from cloud24.report import generate
from v2.checkpoint import source_digest
from v2.runtime import fingerprint, read_json, write_json


def memory_available_gib():
    try:
        values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        return int(values["MemAvailable"].split()[0])/1024**2
    except (OSError, KeyError):
        return 64.


def done(spec):
    if spec["task"] == "train":
        return plan.completed_summary(spec) is not None
    path = Path(spec["directory"])/"test_summary.json"
    if not path.exists():
        return False
    result = read_json(path)
    if result.get("id") != spec["id"] or result.get("source_digest") != source_digest():
        raise RuntimeError(f"Held-out result identity mismatch: {path}")
    return True


class Suite:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.output).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.active, self.specs, self.failures = {}, {}, {}
        self.stop, self.last_report = False, 0.
        self.draining = False
        self.owns_lock = False
        self.gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
        if not self.gpus:
            raise ValueError("At least one GPU is required")
        self.workers = int(args.workers) if args.workers != "auto" else 4*len(self.gpus)
        state_path = self.root/"suite_status.json"
        old = read_json(state_path) if state_path.exists() else None
        if old and not args.resume and not args.dry_run:
            raise RuntimeError("Output exists. Use --resume, or choose a new output directory.")
        if old and old["source_digest"] != source_digest():
            raise RuntimeError("Source changed; use a new output directory, not old checkpoints")
        if old and old.get("quick", False) != args.quick:
            raise RuntimeError("Quick and full studies must use different output directories")
        self.state = old or dict(start_unix=time.time(), start_time=time.strftime("%Y-%m-%d %H:%M:%S %z"),
                                 window_start=time.time(), deadline_unix=time.time()+args.hours*3600,
                                 source_digest=source_digest(), quick=args.quick, phase="initializing", status="running", failed={})
        if args.extend_hours:
            self.state.update(window_start=time.time(), deadline_unix=time.time()+args.extend_hours*3600)
        duration = self.state["deadline_unix"]-self.state["window_start"]
        self.training_end = self.state["deadline_unix"]-min(5400., duration*.1)
        self.failures = dict(self.state.get("failed", {}))
        if args.retry_failed:
            self.failures = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self.request_stop)

    def request_stop(self, *_):
        self.stop = True

    def status(self, **extra):
        self.state.update(extra, pid=os.getpid(), workers=self.workers, gpus=self.gpus,
                          updated_time=time.strftime("%Y-%m-%d %H:%M:%S %z"), failed=self.failures,
                          active=[dict(id=key, pid=value["process"].pid, resource=value["spec"]["resource"],
                                       gpu=value["gpu"], variant=value["spec"]["variant"])
                                  for key, value in self.active.items()])
        write_json(self.root/"suite_status.json", self.state)

    def register(self, tasks):
        for spec in tasks:
            if spec["id"] in self.specs:
                continue
            self.specs[spec["id"]] = spec
            directory = Path(spec["directory"])
            directory.mkdir(parents=True, exist_ok=True)
            target = directory/"task.json"
            if target.exists():
                old = read_json(target)
                if old["id"] != spec["id"]:
                    raise RuntimeError(f"Task path collision: {target}")
            else:
                write_json(target, spec)

    def ready(self, spec):
        if spec["task"] == "train":
            return True
        return (Path(spec["parent"])/"final_eval_summary.json").exists()

    def launch(self, spec, gpu):
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES=gpu or "",
                   MPLCONFIGDIR=str(self.root/".matplotlib"))
        path = Path(spec["directory"])
        stream = (path/"worker.log").open("a", encoding="utf-8")
        process = subprocess.Popen([sys.executable, "-u", "-m", "cloud24.worker", str(path/"task.json")],
                                   cwd=plan.ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        self.active[spec["id"]] = dict(process=process, log=stream, spec=spec, gpu=gpu, start=time.time())
        print(f"START pid={process.pid} gpu={gpu or 'cpu'} {spec['task']} {spec['variant']} N={spec['N']} layout={spec['layout_seed']} seed={spec['seed']}", flush=True)

    def poll(self):
        for identity, item in list(self.active.items()):
            process, spec = item["process"], item["spec"]
            code = process.poll()
            if code is None:
                continue
            item["log"].close()
            del self.active[identity]
            if done(spec):
                print(f"DONE {spec['task']} {spec['variant']} N={spec['N']} seed={spec['seed']}", flush=True)
            elif code != 0 and not self.draining:
                status_path = Path(spec["directory"])/"run_status.json"
                detail = read_json(status_path) if status_path.exists() else {"error": "Process exited before status was written"}
                self.failures[identity] = dict(exit_code=code, directory=spec["directory"], detail=detail)
                print(f"FAILED {spec['variant']}: {spec['directory']} (exit {code})", flush=True)

    def drain(self):
        self.draining = True
        for item in self.active.values():
            item["process"].terminate()
        deadline = time.time()+180
        while self.active and time.time() < deadline:
            self.poll()
            time.sleep(1)
        for identity, item in list(self.active.items()):
            item["process"].kill()
            item["process"].wait()
            item["log"].close()
            path = Path(item["spec"]["directory"])/"run_status.json"
            state = read_json(path) if path.exists() else {}
            state.update(status="interrupted", end_time=time.strftime("%Y-%m-%d %H:%M:%S %z"),
                         error="Grace period expired; resume from last atomic checkpoint")
            write_json(path, state)
            del self.active[identity]
        self.draining = False

    def run_phase(self, phase, tasks, end, workers=None, cpu_workers=None):
        self.register(tasks)
        write_json(self.root/phase/"manifest.json", dict(phase=phase, source_digest=source_digest(),
                   deadline_unix=end, jobs=tasks, validation_only=phase in {"screen", "confirm"}))
        self.status(phase=phase, status="running", planned_tasks=len({s["id"] for s in tasks}))
        limit = workers or self.workers
        cpu_limit = self.args.cpu_workers if cpu_workers is None else cpu_workers
        unique = list({s["id"]: self.specs[s["id"]] for s in tasks}.values())
        while not self.stop and time.time() < end:
            self.poll()
            pending = [s for s in unique if s["id"] not in self.active and s["id"] not in self.failures and not done(s)]
            if not pending and not self.active:
                break
            gpu_counts = {g: sum(item["gpu"] == g for item in self.active.values()) for g in self.gpus}
            gpu_running = sum(s["spec"]["resource"] == "gpu" for s in self.active.values())
            cpu_running = len(self.active)-gpu_running
            launched = False
            for spec in pending:
                if not self.ready(spec):
                    continue
                if memory_available_gib() < self.args.min_free_ram_gib or shutil.disk_usage(self.root).free < 5*1024**3:
                    break
                if spec["resource"] == "cpu":
                    if cpu_running >= cpu_limit:
                        continue
                    self.launch(spec, None)
                    cpu_running += 1
                else:
                    if gpu_running >= limit:
                        continue
                    gpu = min(gpu_counts, key=gpu_counts.get)
                    self.launch(spec, gpu)
                    gpu_counts[gpu] += 1
                    gpu_running += 1
                launched = True
            if not launched and not self.active:
                # Dependencies with a failed parent cannot complete in this invocation.
                blocked = [s for s in pending if not self.ready(s)]
                if len(blocked) == len(pending):
                    break
            if time.time()-self.last_report >= self.args.report_seconds:
                self.status(completed_tasks=sum(done(s) for s in unique))
                generate(self.root)
                self.last_report = time.time()
            time.sleep(.1 if self.args.quick else 3)
        if self.active:
            print(f"STOP phase={phase}; saving episode-boundary checkpoints", flush=True)
            self.drain()
        self.status(completed_tasks=sum(done(s) for s in unique))
        generate(self.root)

    def benchmark(self):
        path = self.root/"benchmark.json"
        if path.exists():
            self.workers = read_json(path)["recommended_workers"]
            return
        counts = [min(4, 4*len(self.gpus)), 4*len(self.gpus)]
        results = []
        for count in sorted(set(counts)):
            tasks = []
            for i in range(count):
                spec = plan.training_job(self.root/"benchmark", "split_bounded", layout=1000+i, seed=i, small=self.args.quick)
                cfg = spec["config"]
                if not self.args.quick:
                    cfg["training"].update(episodes=12, eval_episodes=1, warmup_steps=256,
                                           replay_capacity=10000, checkpoint_every=12)
                # Include candidate count to prevent reuse of the smaller-concurrency benchmark.
                cfg["probe_id"] = f"throughput_{count}"
                spec["id"] = spec["config_hash"] = plan.job_hash(cfg, spec["method"])
                spec["directory"] = str((self.root/"benchmark"/f"workers{count}"/str(i)).resolve())
                tasks.append(spec)
            start = time.time()
            self.run_phase(f"benchmark_{count}", tasks, min(self.training_end, start+1500), workers=count)
            elapsed = time.time()-start
            completed = sum(done(s) for s in tasks)
            results.append(dict(workers=count, wall_seconds=elapsed, completed=completed,
                                aggregate_episodes_per_second=completed*tasks[0]["config"]["training"]["episodes"]/max(elapsed, 1)))
            if self.stop:
                break
        successful = [r for r in results if r["completed"] == r["workers"]]
        if self.stop:
            return
        if not successful:
            raise RuntimeError("No throughput benchmark completed; inspect worker logs before starting a long run")
        chosen = max(successful, key=lambda r: r["aggregate_episodes_per_second"])["workers"]
        self.workers = chosen
        write_json(path, dict(recommended_workers=chosen, candidates=results,
                   note="Short-run aggregate throughput; not a guarantee of long-run completion time."))

    def execute(self):
        if self.args.dry_run:
            screen = plan.screening(self.root, self.args.quick)
            write_json(self.root/"dry_run_plan.json", dict(source_digest=source_digest(), screen=screen,
                       confirmation_example=plan.confirmation(self.root, "split_bounded", self.args.quick),
                       main_if_reference_selected=plan.main_tasks(self.root, "shared_exact", self.args.quick),
                       main_if_candidate_selected=plan.main_tasks(self.root, "split_bounded", self.args.quick),
                       ablations_example=plan.ablation_tasks(self.root, "split_bounded", self.args.quick),
                       tradeoff_example=plan.tradeoff_tasks(self.root, "split_bounded", self.args.quick)))
            print(self.root/"dry_run_plan.json")
            return
        import fcntl
        self.lock = (self.root/"suite.lock").open("a+")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.owns_lock = True
        (self.root/"suite.pid").write_text(str(os.getpid())+"\n")
        if self.args.stage == "export":
            self.export()
            return
        if time.time() >= self.training_end:
            self.status(status="budget_finished")
            self.export()
            return
        if self.args.workers == "auto":
            self.benchmark()
        screen = plan.screening(self.root, self.args.quick)
        selection_path = self.root/"selection_locked.json"
        locked = read_json(selection_path) if selection_path.exists() else None
        stages = plan.PHASES if self.args.stage == "all" else (self.args.stage,)
        for index, stage in enumerate(plan.PHASES):
            if stage not in stages or self.stop or time.time() >= self.training_end:
                continue
            window = self.training_end-self.state["window_start"]
            end = self.state["window_start"]+window*plan.PHASE_FRACTIONS[index] if self.args.stage == "all" else self.training_end
            if time.time() >= end:
                continue
            if stage == "screen":
                if locked:
                    continue
                tasks = screen
            elif stage == "confirm":
                if locked:
                    continue
                choice = plan.select_screen(screen)
                write_json(self.root/"screen_selection.json", choice)
                tasks = plan.confirmation(self.root, choice["candidate"], self.args.quick)
            else:
                if locked is None:
                    choice = plan.select_screen(screen)
                    confirmation = plan.confirmation(self.root, choice["candidate"], self.args.quick)
                    locked = plan.confirm_selection(screen+confirmation, choice["candidate"])
                    locked.update(source_digest=source_digest(), locked_unix=time.time(),
                                  validation_stream="700000 + 100*N + layout_index",
                                  held_out_stream="900000 + 100*N + layout_index")
                    write_json(selection_path, locked)
                selected = locked["selected"]
                tasks = (plan.main_tasks(self.root, selected, self.args.quick, extension=stage == "extension")
                         if stage in {"main", "extension"} else
                         plan.ablation_tasks(self.root, selected, self.args.quick) if stage == "ablations" else
                         plan.tradeoff_tasks(self.root, selected, self.args.quick))
                if self.args.stage == "all" and stage in {"ablations", "tradeoff", "extension"}:
                    core = plan.main_tasks(self.root, selected, self.args.quick)
                    pending_core = [job for job in core if not done(job) and job["id"] not in self.failures]
                    # Do not trade missing main-comparison replicates for optional mechanisms or scales.
                    tasks = pending_core + tasks
            self.run_phase(stage, tasks, end)
        status = "interrupted" if self.stop else "budget_finished" if time.time() >= self.training_end else "planned_stages_finished"
        self.status(status=status)
        self.export()

    def export(self):
        generate(self.root, final=True)
        write_json(self.root/"runtime_manifest.json", dict(python=sys.version, source_digest=source_digest(),
                   output=str(self.root), args=vars(self.args), deadline_unix=self.state["deadline_unix"],
                   gpu_command="CUDA_VISIBLE_DEVICES=<one assigned GPU>; device=cuda:0 per worker",
                   no_action_repair=True, gamma=0., energy_ref_J=.02))
        # This compact export excludes large replay snapshots, but keeps final models and all observations.
        archive = self.root.parent/(self.root.name+"_results.tar.gz")
        with tarfile.open(archive, "w:gz", compresslevel=1) as stream:
            for path in self.root.rglob("*"):
                if path.is_file() and path.name not in {"checkpoint.pt", "suite.lock"} and ".matplotlib" not in path.parts:
                    stream.add(path, arcname=str(Path(self.root.name)/path.relative_to(self.root)))
        print(f"RESULTS {self.root}\nEXPORT {archive}", flush=True)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/cloud24")
    parser.add_argument("--hours", type=float, default=24.)
    parser.add_argument("--extend-hours", type=float, default=0.)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--workers", default="auto", help="auto benchmarks 4 vs 8 on two GPUs; or an integer")
    parser.add_argument("--cpu-workers", type=int, default=2)
    parser.add_argument("--min-free-ram-gib", type=float, default=12.)
    parser.add_argument("--stage", choices=("all", *plan.PHASES, "export"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Tiny protocol run; results stay in a separate output directory")
    parser.add_argument("--report-seconds", type=float, default=1800.)
    args = parser.parse_args()
    if args.hours <= 0 or args.extend_hours < 0 or args.cpu_workers < 0:
        parser.error("Invalid time or worker limits")
    if args.workers != "auto" and int(args.workers) <= 0:
        parser.error("--workers must be auto or positive")
    if args.quick and args.output == "outputs/cloud24":
        args.output = "outputs/quick"
    return args


def main():
    suite = Suite(arguments())
    try:
        suite.execute()
    except BaseException:
        if suite.owns_lock:
            suite.drain()
            import traceback
            suite.status(status="failed", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
