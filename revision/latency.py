"""E6 decision latency on one CPU thread (actor forward pass plus action decoding).

Latency does not depend on trained weights, so N = 50, 70, 200 use freshly initialized networks of
the deployed architectures; the trained N = 70 SFR-SAC actor is timed as a check when available.
The per-slot optimizers' decision times are measured inside their own evaluation tasks.
"""
from __future__ import annotations

import platform
import time
from pathlib import Path

import numpy as np
import torch

from offloading.runtime import execution_action
from revision import variants as V
from v2.checkpoint import source_digest
from v2.runtime import load_agent, make_agent, make_env, read_json, write_json


def cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def time_policy(agent, env, method, warmup, reps):
    states = [env.reset()]
    for _ in range(31):
        state, _, done, _ = env.step(env.action_space.sample())
        states.append(env.reset() if done else state)
    with torch.inference_mode():
        for i in range(warmup):
            execution_action(agent.select_action(states[i % 32], deterministic=True), env, method)
        samples = np.empty(reps)
        for i in range(reps):
            start = time.perf_counter_ns()
            execution_action(agent.select_action(states[i % 32], deterministic=True), env, method)
            samples[i] = (time.perf_counter_ns() - start) / 1e6
    return dict(median_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples, 95)),
                p99_ms=float(np.percentile(samples, 99)))


def run(spec, out):
    torch.set_num_threads(1)
    torch.set_flush_denormal(True)
    rows = []
    warmup, reps = spec["warmup"], spec["reps"]
    for variant, method in (("sfr", "safe"), ("td3", "td3_lag")):
        for n in spec["sizes"]:
            cfg = V.configure(variant, n=n, small=spec.get("small", False))
            cfg["execution"] = dict(device="cpu", require_cuda=False, tf32=False)
            np.random.seed(0)
            torch.manual_seed(0)
            env = make_env(cfg, n, 0)
            agent = make_agent(cfg, method, env, "cpu")
            agent.actor.eval()
            row = dict(variant=variant, N=n, weights="initialized", threads=1,
                       parameters_actor=sum(p.numel() for p in agent.actor.parameters()),
                       state_dim=env.state_dim, action_dim=env.action_dim)
            row.update(time_policy(agent, env, method, warmup, reps))
            rows.append(row)
    trained = spec.get("trained_parent")
    if trained and (Path(trained) / "model.pt").exists():
        cfg = read_json(Path(trained) / "job.json")["config"]
        agent, method, n, seed = load_agent(Path(trained) / "model.pt", cfg, device="cpu")
        cfg = dict(cfg, execution=dict(device="cpu", require_cuda=False, tf32=False))
        row = dict(variant="sfr", N=n, weights="trained", threads=1,
                   parameters_actor=sum(p.numel() for p in agent.actor.parameters()))
        row.update(time_policy(agent, make_env(cfg, n, seed), method, warmup, reps))
        rows.append(row)
    summary = dict(rows=rows, cpu=cpu_name(), torch=torch.__version__, python=platform.python_version())
    write_json(Path(out) / "test_summary.json", dict(id=spec["id"], source_digest=source_digest(),
                                                     spec={k: v for k, v in spec.items() if k != "config"},
                                                     summary=summary, split="latency"))
    return summary
