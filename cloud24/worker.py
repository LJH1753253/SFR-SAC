"""One isolated training, validation, or held-out evaluation process."""

import copy
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np
import torch

from cloud24.plan import completed_summary
from v2.checkpoint import source_digest
from v2.runtime import (evaluate_agent, evaluate_policy, load_agent, make_agent, make_env,
                        make_heuristic, read_json, run_job, seed_all, write_json)


def nonlearning_training(spec, out):
    from v2.runtime import run_heuristic, run_lyapunov
    cfg, method = spec["config"], spec["method"]
    function = run_lyapunov if method == "lyapunov" else run_heuristic
    summary = function(cfg, spec["N"], spec["seed"], out) if method == "lyapunov" else function(
        cfg, method, spec["N"], spec["seed"], out)
    write_json(out/"final_eval_summary.json", dict(config_hash=spec["config_hash"],
               source_digest=source_digest(), config=cfg, job={k: spec[k] for k in ("variant", "method", "N", "seed")},
               summary=summary))


def heldout(spec, out):
    cfg = copy.deepcopy(spec["config"])
    n, seed, method = spec["N"], spec["seed"], spec["method"]
    parent = Path(spec["parent"])
    source = read_json(parent/"final_eval_summary.json")
    if source["config_hash"] != spec["config_hash"] or source["source_digest"] != source_digest():
        raise RuntimeError("Parent result identity mismatch")
    # The per-method identity ignores inactive algorithms; load the exact saved configuration.
    cfg = read_json(parent/"job.json")["config"]
    device = cfg["execution"]["device"]
    torch.set_num_threads(1)
    original = source["summary"]
    if method in {"safe", "penalty_sac", "vanilla", "td3_lag", "cpo"}:
        agent, _, _, _ = load_agent(parent/"model.pt", cfg, device=device)
        cfg["evaluation_rng_seed"] = spec["test_rng_seed"]
        summary = evaluate_agent(cfg, method, agent, n, seed, spec["test_episodes"], details_dir=out)
        state = make_env(cfg, n, seed).reset()
        for _ in range(20):
            agent.select_action(state, deterministic=True)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(200):
            agent.select_action(state, deterministic=True)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        summary["policy_inference_ms_batch1"] = 1000*(time.perf_counter()-start)/200
        summary["parameters_actor"] = sum(p.numel() for p in agent.actor.parameters())
    elif method == "lyapunov":
        seed_all(seed)
        agent = make_agent(cfg, method, make_env(cfg, n, seed))
        # Recreate the virtual queue after the same declared warm-up, without learning an actor.
        env = make_env(cfg, n, seed)
        from offloading.runtime import execution_action
        for _ in range(cfg["training"]["episodes"]):
            state, done = env.reset(), False
            while not done:
                state, _, done, info = env.step(execution_action(agent.select_action(state), env, method))
                agent.update_virtual_queue(info["cost"], info)
        cfg["evaluation_rng_seed"] = spec["test_rng_seed"]
        summary = evaluate_agent(cfg, method, agent, n, seed, spec["test_episodes"], details_dir=out)
    else:
        keys = {key[5:]: value for key, value in original.items() if key.startswith("knob_")}
        policy = make_heuristic(method, keys)
        summary = evaluate_policy(cfg, n, seed, lambda env, state: policy.act(env), spec["test_episodes"],
                                  spec["test_rng_seed"], details_dir=out)
        summary.update({"knob_"+k: v for k, v in keys.items()})
    for key in ("final_lambda", "final_alpha", "mean_lambda_last_window", "lambda_std_last_window",
                "train_seconds", "evaluation_seconds", "training_steps", "actor_architecture",
                "cost_critic_architecture", "reward_critic_architecture"):
        if key in original:
            summary[key] = original[key]
    write_json(out/"test_summary.json", dict(id=spec["id"], source_digest=source_digest(),
               spec={k: v for k, v in spec.items() if k != "config"}, summary=summary,
               configuration=cfg, split="held_out_test", test_rng_seed=spec["test_rng_seed"]))


def main(path):
    spec = read_json(path)
    out = Path(spec["directory"])
    out.mkdir(parents=True, exist_ok=True)
    status = dict(status="running", start_time=time.strftime("%Y-%m-%d %H:%M:%S %z"),
                  end_time=None, command=[sys.executable, "-m", "cloud24.worker", str(path)],
                  pid=os.getpid(), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""), error=None)
    if (out/"run_status.json").exists():
        write_json(out/"attempt_history"/f"{time.time_ns()}_{os.getpid()}.json", read_json(out/"run_status.json"))
    write_json(out/"run_status.json", status)
    try:
        expected_device = spec["resource"]
        if expected_device == "gpu" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; refusing CPU fallback")
        if expected_device == "gpu":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        write_json(out/"hardware.json", dict(torch=torch.__version__, cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                   source_digest=source_digest(), train_seed=spec["seed"], layout_seed=spec["layout_seed"]))
        if spec["task"] == "test":
            heldout(spec, out)
        else:
            write_json(out/"job.json", spec)
            if spec["kind"] == "learner":
                run_job(out/"job.json")
                if completed_summary(spec) is None:
                    status["status"] = "interrupted"
            else:
                nonlearning_training(spec, out)
        if status["status"] == "running":
            status["status"] = "done"
    except BaseException:
        status.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        status["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
        write_json(out/"run_status.json", status)


if __name__ == "__main__":
    main(sys.argv[1])
