"""Configuration, construction, training loops, and evaluation for the v2 experiments."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
import traceback
import signal
from functools import partial
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.safe_sac_agent import SafeSACAgent, VanillaSACAgent  # noqa: E402
from agents.td3_agent import TD3Agent, TD3Config  # noqa: E402
from offloading.runtime import execution_action  # noqa: E402
from v2 import baselines as B  # noqa: E402
from v2.cpo import CPOAgent  # noqa: E402
from v2.env import HeteroOffloadingEnv  # noqa: E402
from v2.metrics import EvalAccumulator  # noqa: E402
from v2.replay import RingReplay  # noqa: E402
from v2.recency_replay import RecencyReplay  # noqa: E402
from v2.checkpoint import agent_state, atomic_save, restore_agent, source_digest  # noqa: E402

LEARNERS = ("safe", "penalty_sac", "td3_lag", "vanilla", "cpo")
HEURISTICS = ("lyapunov", "const_ratio", "local_first", "mb_greedy")
METHODS = LEARNERS + HEURISTICS
ACTOR_DIAGNOSTIC_FIELDS = (
    "actor_grad_norm_mean", "actor_grad_norm_max", "actor_clipped_fraction",
    "actor_updates_in_episode", "critic_updates_in_episode", "actor_entropy_term_mean",
    "actor_energy_term_mean", "actor_constraint_term_mean", "cost_critic_disagreement_mean",
    "ap_entropy_mean", "ap_max_prob_mean",
    "entropy_driver_mean", "continuous_log_prob_per_dim_mean", "categorical_log_prob_per_device_mean",
    "replay_age_steps_mean", "replay_recent_fraction_mean",
)
AP_DIAGNOSTIC_FIELDS = ("train_ap_load_cv", "train_best_ap_percent")


# ----------------------------------------------------------------------------- utilities
def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, default=_jsonable) + "\n", encoding="utf-8")
    for attempt in range(20):  # Windows: antivirus or sync clients may hold the target briefly
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.5 * (attempt + 1))
    tmp.replace(path)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def fingerprint(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=_jsonable).encode()).hexdigest()[:16]


def seed_all(seed):
    # Flush subnormal floats to zero. Without this, matrix products slow down 100x once tiny
    # gradients appear, and episodes become 3-4x slower after ~100 episodes of training.
    torch.set_flush_denormal(True)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_csv(path, row, first):
    with Path(path).open("w" if first else "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if first:
            w.writeheader()
        w.writerow(row)


# ----------------------------------------------------------------------------- construction
def make_env(cfg, n, seed):
    env_cfg = dict(cfg["environment"])
    env_cfg["num_devices"] = int(n)
    env_cfg["layout_seed"] = int(cfg.get("layout_seed", int(cfg["layout_seed_base"]) + int(seed)))
    return HeteroOffloadingEnv(**env_cfg)


def agent_gamma(cfg, name):
    g = cfg["algorithms"].get(name, {}).get("gamma")
    return float(cfg["training"]["gamma"] if g is None else g)


def reward_scale(cfg, name):
    rs = cfg["training"]["reward_scale"]
    if rs == "auto":
        g = agent_gamma(cfg, name)
        return 1.0 - g if g < 1.0 else 1.0
    return float(rs)


def training_reward(cfg, name, reward, cost):
    value = reward
    if name == "penalty_sac":
        value = reward - float(cfg["algorithms"]["penalty_sac"]["penalty_sac_beta"]) * cost
    return reward_scale(cfg, name) * value


def assumed_parameters(settings, env):
    """Physical constants given to structured actors/critics. They equal the simulator's unless a
    model-mismatch study overrides some of them (`model_assumptions`)."""
    params = dict(env.params)
    params.update(settings.get("model_assumptions") or {})
    return params


def policy_factory(settings, env):
    architecture = settings.get("actor_architecture", "mlp")
    if architecture == "mlp":
        return None
    from agents.structured_actor import PhysicalFeatureMLPActor, ServiceFlowActor
    if architecture == "feature_mlp":
        return partial(PhysicalFeatureMLPActor, parameters=assumed_parameters(settings, env))
    if architecture not in {"service_flow", "split_service_flow"}:
        raise ValueError("Unknown actor_architecture")
    factory, extra = ServiceFlowActor, {}
    if architecture == "split_service_flow":
        from agents.cloud_actor import SplitServiceFlowActor
        factory = SplitServiceFlowActor
        extra["ap_residual_bound"] = settings.get("ap_residual_bound")
        if settings.get("actor_ap_rule", "learned") != "learned":
            extra["ap_rule"] = settings["actor_ap_rule"]
    return partial(factory, parameters=assumed_parameters(settings, env), **extra,
                   hidden=int(settings.get("actor_hidden", 64)),
                   embedding_dim=int(settings.get("actor_embedding_dim", 8)),
                   use_physics_features=settings.get("actor_use_physics_features", True),
                   use_pooled_context=settings.get("actor_use_pooled_context", True),
                   use_device_embedding=settings.get("actor_use_device_embedding", True))


def make_agent(cfg, name, env, device="cpu"):
    a = cfg["algorithms"][name]
    d = float(cfg["cost_limit"])
    act = cfg.get("actor", {})
    ap_prior = None
    if act.get("ap_prior", False):
        ap_prior = {"start": env.link_state_offset, "init_scale": float(act.get("ap_prior_init_scale", 25.0)),
                    "learnable": bool(act.get("ap_prior_learnable", True))}
    common = dict(state_dim=env.state_dim, action_dim=env.action_dim, device=device, max_action=1.0,
                  num_devices=env.N, num_rsu=env.R, ap_projection_mode="hybrid_ap_gumbel", ap_prior=ap_prior)
    dual = lambda: dict(  # noqa: E731
        cost_limit=a.get("training_cost_limit", d), lambda_lr=a["lambda_lr"], lambda_mode=a["lambda_mode"],
        fixed_lambda=a["fixed_lambda"], initial_lambda=a["initial_lambda"],
        slow_lambda_factor=a["slow_lambda_factor"], cost_coef=a["cost_coef"],
        update_lambda_in_batch=a["lambda_update_granularity"] == "batch",
        cost_critic_target_mode=a["cost_critic_target_mode"])
    if name == "cpo":
        return CPOAgent(env.state_dim, env.N, env.R, cost_limit=d, gamma=agent_gamma(cfg, name),
                        lam=a["gae_lambda"], cost_lam=a["gae_lambda"],
                        cost_gamma=agent_gamma(cfg, name) if a.get("cost_gamma") is None else a["cost_gamma"],
                        target_kl=a["target_kl"], damping=a["damping"], cg_iters=a["cg_iters"],
                        backtrack_iters=a["backtrack_iters"], backtrack_coeff=a["backtrack_coeff"],
                        vf_lr=a["vf_lr"], vf_iters=a["vf_iters"], vf_batch=a["vf_batch"],
                        steps_per_update=a["steps_per_update"], device=device, ap_prior=ap_prior,
                        actor_factory=policy_factory(a, env), fisher_fraction=a.get("fisher_fraction", 1.0))
    if name == "td3_lag":
        td3 = TD3Config(lr=a["td3_lr"], tau=a["td3_tau"], policy_noise=a["td3_policy_noise"],
                        exploration_noise=a["td3_exploration_noise"],
                        exploration_noise_rel_scale=a["td3_exploration_noise_rel_scale"],
                        exploration_noise_ap_scale=a["td3_exploration_noise_ap_scale"],
                        policy_noise_rel_scale=a["td3_policy_noise_rel_scale"],
                        policy_noise_ap_scale=a["td3_policy_noise_ap_scale"])
        td3.gamma = agent_gamma(cfg, name)
        structured = {}
        if a.get("critic_architecture", "mlp") == "service_flow":
            # Same structured reward/cost critics as SFR-SAC (revision fair-critic study).
            from agents.structured_cost_critic import ServiceFlowCostCritic
            from agents.structured_energy_critic import ServiceFlowEnergyCritic
            params = assumed_parameters(a, env)
            structured.update(
                critic_factory=partial(ServiceFlowEnergyCritic, num_devices=env.N, num_rsu=env.R, parameters=params,
                                       hidden=int(a.get("energy_hidden", 64)), reward_scale=reward_scale(cfg, name)),
                cost_critic_factory=partial(ServiceFlowCostCritic, num_devices=env.N, num_rsu=env.R,
                                            parameters=params, hidden=int(a.get("cost_delivery_hidden", 64))),
                energy_component_loss_weight=a.get("energy_component_loss_weight", 1.0),
                cost_delivery_loss_weight=a.get("cost_delivery_loss_weight", 1.0))
        elif a.get("critic_architecture", "mlp") != "mlp":
            raise ValueError("Unknown td3 critic_architecture")
        if a.get("actor_architecture", "mlp") != "mlp":
            structured["policy_factory"] = policy_factory(a, env)
        return TD3Agent(**common, **dual(), cfg=td3, use_lagrangian=True, **structured)
    if name == "lyapunov":
        return B.LyapunovV2(
            env, num_devices=env.N, num_rsu=env.R, tx_power=0.5 * (env.tx_power_min + env.tx_power_max),
            tx_power_control="action", tx_power_min=env.tx_power_min, tx_power_max=env.tx_power_max,
            task_model="oneshot", mode=a["lyapunov_mode"], queue_target=a["lyapunov_queue_target"],
            max_offload=a["lyapunov_max_offload"], access_margin=a["lyapunov_access_margin"],
            edge_margin=a["lyapunov_edge_margin"], v_energy=a["lyapunov_v_energy"],
            eta_delivery=a["lyapunov_eta_delivery"], eta_queue=a["lyapunov_eta_queue"],
            delivery_limit=d, queue_limit=a["lyapunov_queue_limit"], initial_z=a["lyapunov_initial_z"])
    sac = dict(gamma=agent_gamma(cfg, name), alpha=a["fixed_alpha"], auto_alpha=bool(a["auto_alpha"]),
               alpha_lr=a["alpha_lr"], target_entropy=-float(env.N), action_mode="hybrid_ap_gumbel",
               gumbel_tau=a["gumbel_tau"], gumbel_tau_min=a["gumbel_tau_min"],
               ap_entropy_weight=a["ap_entropy_weight"],
               ap_entropy_mode=a.get("ap_entropy_mode", "legacy_sampled"),
               matched_entropy_temperature=a.get("matched_entropy_temperature", False),
               freeze_ap_policy=a.get("freeze_ap_policy", False))
    if name == "safe":
        architecture = a.get("cost_critic_architecture", "mlp")
        if architecture not in {"mlp", "service_flow"}:
            raise ValueError("Unknown cost_critic_architecture")
        cost_factory = None
        if architecture == "service_flow":
            from agents.structured_cost_critic import ServiceFlowCostCritic
            cost_factory = partial(ServiceFlowCostCritic, num_devices=env.N, num_rsu=env.R,
                                   parameters=assumed_parameters(a, env),
                                   hidden=int(a.get("cost_delivery_hidden", 64)))
        reward_architecture = a.get("reward_critic_architecture", "mlp")
        if reward_architecture not in {"mlp", "service_flow"}:
            raise ValueError("Unknown reward_critic_architecture")
        reward_factory = None
        if reward_architecture == "service_flow":
            from agents.structured_energy_critic import ServiceFlowEnergyCritic
            reward_factory = partial(ServiceFlowEnergyCritic, num_devices=env.N, num_rsu=env.R,
                                     parameters=assumed_parameters(a, env),
                                     hidden=int(a.get("energy_hidden", 64)),
                                     reward_scale=reward_scale(cfg, name))
        actor_factory = policy_factory(a, env)
        agent = SafeSACAgent(**common, **sac, **dual(), lr=a["safe_lr"], tau=a["safe_tau"],
                             penalty_mode=a.get("penalty_mode", "hinge"), cost_critic_factory=cost_factory,
                             cost_delivery_loss_weight=a.get("cost_delivery_loss_weight", 1.0),
                             reward_critic_factory=reward_factory,
                             energy_component_loss_weight=a.get("energy_component_loss_weight", 1.0),
                             dual_cost_source=a.get("dual_cost_source", "replay_cost"),
                             dual_risk_mode=a.get("dual_risk_mode", "mean"),
                             excess_budget=a.get("excess_budget", 0.0002), actor_factory=actor_factory,
                             actor_grad_clip=a.get("actor_grad_clip"),
                             actor_update_interval=a.get("actor_update_interval", 1),
                             freeze_alpha=a.get("freeze_alpha", False),
                             ap_gradient_mode=a.get("ap_gradient_mode", "straight_through"),
                             ap_value_devices=a.get("ap_value_devices", 4),
                             ap_value_chunk=a.get("ap_value_chunk", 512))
    elif name in ("vanilla", "penalty_sac"):
        agent = VanillaSACAgent(**common, **sac, lr=a["vanilla_lr"], tau=a["vanilla_tau"],
                                actor_factory=policy_factory(a, env))
    else:
        raise ValueError(f"Unknown learner {name}")
    return agent


def make_heuristic(name, knob):
    if name == "const_ratio":
        return B.ConstRatio(x=knob["x"], p=knob["p"])
    if name == "local_first":
        return B.LocalFirst()
    if name == "mb_greedy":
        return B.MBGreedy(slack=knob["slack"])
    raise ValueError(name)


MODULES = ("actor", "critic_1", "critic_2", "cost_critic_1", "cost_critic_2", "q1", "q2", "qc1",
           "qc2", "vf", "vc")


def save_agent(path, agent, cfg, name, n, seed):
    mods = {k: getattr(agent, k).state_dict() for k in MODULES if hasattr(agent, k)}
    payload = {"agent": name, "N": n, "seed": seed, "config_hash": fingerprint(cfg), "modules": mods,
               "source_digest": source_digest(),
               "scalars": {k: float(getattr(agent, k)) for k in ("alpha", "lambda_val") if hasattr(agent, k)}}
    atomic_save(path, payload)
    atomic_save(Path(path).with_name("actor.pt"), {**payload, "modules": {"actor": mods["actor"]}})


def load_agent(path, cfg, device="cpu", check_source=True):
    ck = torch.load(path, map_location=device, weights_only=True)
    if ck["config_hash"] != fingerprint(cfg) or (check_source and ck.get("source_digest") != source_digest()):
        raise ValueError("Model source/configuration mismatch")
    env = make_env(cfg, ck["N"], ck["seed"])
    agent = make_agent(cfg, ck["agent"], env, device)
    for k, sd in ck["modules"].items():
        getattr(agent, k).load_state_dict(sd)
    for key, value in ck.get("scalars", {}).items():
        setattr(agent, key, value)
    if torch.is_tensor(getattr(agent, "lam", None)):  # SAC dual variable (CPO's `lam` is its GAE lambda)
        agent.lam.fill_(agent.lambda_val)
    return agent, ck["agent"], ck["N"], ck["seed"]


# ----------------------------------------------------------------------------- evaluation
def eval_seed(n, seed, base=99000):
    return int(base) + 10 * int(n) + int(seed)


def val_seed(n, seed, base=88000):
    return int(base) + 10 * int(n) + int(seed)


def evaluate_policy(cfg, n, seed, act_fn, episodes, rng_seed, post_step=None, details_dir=None):
    seed_all(rng_seed)
    env = make_env(cfg, n, seed)
    acc = EvalAccumulator(env.N, cfg["cost_limit"])
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            state, _, done, info = env.step(act_fn(env, state))
            acc.add(info)
            if post_step is not None:
                post_step(info)
        acc.end_episode()
    if details_dir is not None:
        details_dir = Path(details_dir)
        details_dir.mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(acc.rows):
            append_csv(details_dir / "eval_episodes.csv", {"episode": i, **row}, i == 0)
        for i in range(n):
            append_csv(details_dir / "per_device.csv",
                       dict(device=i, failed_bits=acc.fail[i], workload_bits=acc.bits[i],
                            failure_percent=100.0 * acc.fail[i] / max(acc.bits[i], 1e-12)), i == 0)
        for i, value in enumerate(acc.slot_costs):
            append_csv(details_dir / "eval_slots.csv",
                       dict(slot=i, cost=value, budget=cfg["cost_limit"], feasible=value <= cfg["cost_limit"]), i == 0)
    return acc.summary()


def evaluate_agent(cfg, name, agent, n, seed, episodes, details_dir=None):
    def act(env, state):
        return execution_action(agent.select_action(state, deterministic=True), env, name)
    post = None
    if name == "lyapunov":  # drift-plus-penalty is an online rule: its virtual queue keeps running
        post = lambda info: agent.update_virtual_queue(info["cost"], info)  # noqa: E731
    return evaluate_policy(cfg, n, seed, act, episodes, cfg.get("evaluation_rng_seed", eval_seed(n, seed, cfg.get("evaluation_seed_base", 99000))), post_step=post,
                           details_dir=details_dir)


# ----------------------------------------------------------------------------- jobs
def run_learner(cfg, name, n, seed, out, device="cpu", resume=True, stop_after=None, stop_requested=None):
    from v2.plan import job_hash

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tr = cfg["training"]
    if cfg.get("execution", {}).get("require_cuda", False) and (
        not str(device).startswith("cuda") or not torch.cuda.is_available()
    ):
        raise RuntimeError("CUDA required; refusing CPU fallback")
    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(int(tr["torch_threads"]))
    seed_all(seed)
    env = make_env(cfg, n, seed)
    env.action_space.seed(seed)
    agent = make_agent(cfg, name, env, device)
    device_map = {key: str(next(getattr(agent, key).parameters()).device)
                  for key in ("actor", "critic_1", "critic_2", "cost_critic_1", "cost_critic_2")
                  if hasattr(agent, key)}
    if cfg.get("execution", {}).get("require_cuda", False) and any(
        not value.startswith("cuda") for value in device_map.values()
    ):
        raise RuntimeError("A required network is not on CUDA")
    write_json(out / "device_check.json", device_map)
    gran = cfg["algorithms"].get(name, {}).get("lambda_update_granularity", "batch")
    t0 = time.perf_counter()
    identity = job_hash({**cfg, "method": name, "N": n, "seed": seed}, name)
    checkpoint_path = out / "checkpoint.pt"
    replay_factory = RecencyReplay if name == "safe" else RingReplay
    replay_options = dict(recent_fraction=cfg["algorithms"][name].get("replay_recent_fraction", 0.),
                          recent_window=cfg["algorithms"][name].get("replay_recent_window", 20000)) if name == "safe" else {}
    replay = None if name == "cpo" else replay_factory(
        tr["replay_capacity"], env.state_dim, env.action_dim, seed,
        delivery_dim=env.N if getattr(agent, "structured_cost_enabled", False) else 0,
        energy_dim=env.N if getattr(agent, "structured_energy_enabled", False) else 0, **replay_options)
    start_episode, history = 0, []
    if resume and not checkpoint_path.exists() and cfg.get("intervention"):
        from v2.intervention import fork_checkpoint
        fork_checkpoint(cfg, checkpoint_path, identity, device)
    if resume and not checkpoint_path.exists() and cfg.get("continuation"):
        parent = Path(cfg["continuation"])
        previous_job = read_json(parent / "job.json")
        previous_cfg = previous_job["config"]
        comparable = copy.deepcopy(cfg)
        comparable.pop("continuation", None)
        comparable["training"]["episodes"] = previous_cfg["training"]["episodes"]
        previous_comparable = copy.deepcopy(previous_cfg)
        previous_comparable.pop("continuation", None)
        if comparable != previous_comparable or tr["episodes"] <= previous_cfg["training"]["episodes"]:
            raise ValueError("Continuation changes more than the training horizon")
        saved = torch.load(parent / "checkpoint.pt", map_location=device, weights_only=False)
        if saved.get("complete", False) or saved["identity"] != previous_job["config_hash"]:
            raise ValueError("Continuation requires a matching full training checkpoint")
        if previous_job["config_hash"] != job_hash(previous_cfg, name):
            raise ValueError("Continuation source/configuration mismatch")
        saved["identity"] = identity
        atomic_save(checkpoint_path, saved)
        write_json(out / "continuation.json", dict(parent=str(parent), resumed_episode=saved["episode"],
                   old_identity=previous_job["config_hash"], new_identity=identity))
    if resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if saved["identity"] != identity:
            raise ValueError("Checkpoint source/configuration mismatch")
        if not saved.get("complete", False):
            restore_agent(agent, saved["agent"])
            if replay is not None:
                replay.load_state_dict(saved["replay"])
            start_episode, history = saved["episode"], saved["history"]
            random.setstate(saved["python_rng"])
            np.random.set_state(saved["numpy_rng"])
            torch.set_rng_state(saved["torch_rng"].cpu())
            if str(device).startswith("cuda") and saved["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
            t0 -= saved["elapsed_seconds"]
            for i, row in enumerate(history):
                append_csv(out / "training.csv", row, i == 0)

    def checkpoint(episode):
        atomic_save(checkpoint_path, dict(
            identity=identity, episode=episode, history=history, agent=agent_state(agent),
            replay=replay.state_dict() if replay is not None else None,
            python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else None,
            elapsed_seconds=time.perf_counter() - t0,
        ))

    def finish_episode(ep, rows, extra=None):
        history.append(_log_episode(out, ep, rows, agent, t0, extra=extra))
        stopping = bool(stop_requested and stop_requested())
        if ep % int(tr.get("checkpoint_every", 25)) == 0 or ep == tr["episodes"] or ep == stop_after or stopping:
            checkpoint(ep)
        if name == "safe" and ep in tr.get("snapshot_episodes", ()):
            save_agent(out / f"model_ep{ep}.pt", agent, cfg, name, n, seed)
        return ep == stop_after or stopping

    def store_transition(state, action, reward, info, nxt):
        if not np.isfinite(action).all() or not np.isfinite([reward, info["cost"], info["energy"]]).all():
            raise RuntimeError("Non-finite action, reward, cost, or energy")
        label = None
        if getattr(agent, "structured_cost_enabled", False):
            label = info["delivered_bits_vec"] / np.maximum(info["task_bits_vec"], 1e-12)
        energy = None
        if getattr(agent, "structured_energy_enabled", False):
            energy = info["energy_components_vec"] / env.energy_ref
        replay.push(state, action, training_reward(cfg, name, reward, info["cost"]), info["cost"],
                    nxt, False, delivery_fraction=label, energy_components=energy)

    if name == "cpo":
        for ep in range(start_episode + 1, tr["episodes"] + 1):
            rows, done = [], False
            state = env.reset()
            info_last = {}
            while not done:
                action, z, u = agent.sample(state)
                nxt, reward, done, info = env.step(action)
                agent.store(state, z, u, training_reward(cfg, name, reward, info["cost"]), info["cost"], done)
                if agent.ready():
                    info_last = agent.update()
                rows.append((info["cost"], info["energy"], info["executed_offload_ratio_mean"], info["tx_power_mean"]))
                state = nxt
            if finish_episode(ep, rows, info_last):
                return None
        if agent.buf:
            agent.update()
    else:
        if start_episode == 0:
            state = env.reset()
            for _ in range(tr["warmup_steps"]):
                action = execution_action(env.action_space.sample(), env, "safe")
                nxt, reward, done, info = env.step(action)
                store_transition(state, action, reward, info, nxt)
                state = env.reset() if done else nxt
        for ep in range(start_episode + 1, tr["episodes"] + 1):
            env.update_training_progress(ep)
            state = env.reset()
            rows, costs = [], []
            update_info = {}
            episode_updates = []
            ap_measurements = []
            for step in range(1, env.max_ep_steps + 1):
                action = execution_action(agent.select_action(state), env, name)
                nxt, reward, done, info = env.step(action)
                store_transition(state, action, reward, info, nxt)
                if name in ("safe", "td3_lag") and gran == "step":
                    agent.update_lambda(info["cost"])
                if len(replay) >= tr["batch_size"] and step % tr["update_freq"] == 0:
                    update_info = agent.update(replay, batch_size=tr["batch_size"])
                    if name == "safe" and cfg["algorithms"][name].get("actor_stability_diagnostics", False):
                        episode_updates.append(update_info)
                rows.append((info["cost"], info["energy"], info["executed_offload_ratio_mean"], info["tx_power_mean"]))
                costs.append(info["cost"])
                if cfg["algorithms"].get(name, {}).get("ap_branch_diagnostics", False):
                    counts = info["ap_counts"]
                    ap_measurements.append((float(np.std(counts)/max(np.mean(counts),1e-9)),
                        float(100*np.mean(info["ap_index_vec"] == info["best_ap_vec"]))))
                state = nxt
                if done:
                    break
            if name in ("safe", "td3_lag") and gran == "episode":
                agent.update_lambda(float(np.mean(costs)))
            extra = None
            if getattr(agent, "structured_cost_enabled", False):
                extra = {key: float(update_info.get(key, 0.0)) for key in ("loss_qc", "loss_delivery")}
            if getattr(agent, "structured_energy_enabled", False):
                extra = dict(extra or {}, loss_energy_components=float(update_info.get("loss_energy_components", 0.0)))
            if cfg["algorithms"].get(name, {}).get("constraint_diagnostics", False):
                extra = dict(extra or {}, **{key: update_info.get(key) for key in (
                    "dual_statistic", "dual_budget", "dual_replay_cost_mean", "cost_pi_mean")})
            if name == "safe" and cfg["algorithms"][name].get("actor_stability_diagnostics", False):
                changed = [u for u in episode_updates if u["actor_updated"]]
                norms = [u["actor_grad_norm"] for u in changed]
                measurements = dict(
                    actor_grad_norm_mean=float(np.mean(norms)) if norms else None,
                    actor_grad_norm_max=max(norms) if norms else None,
                    actor_clipped_fraction=float(np.mean([u["actor_clipped"] for u in changed])) if changed else None,
                    actor_updates_in_episode=len(changed), critic_updates_in_episode=len(episode_updates),
                )
                for key in ("actor_entropy_term", "actor_energy_term", "actor_constraint_term",
                            "cost_critic_disagreement", "ap_entropy", "ap_max_prob", "entropy_driver",
                            "continuous_log_prob_per_dim", "categorical_log_prob_per_device",
                            "replay_age_steps", "replay_recent_fraction"):
                    measurements[key+"_mean"] = float(np.mean([u[key] for u in episode_updates])) if episode_updates else None
                extra = dict(extra or {}, **measurements)
            if cfg["algorithms"].get(name, {}).get("ap_branch_diagnostics", False):
                values = np.mean(ap_measurements, axis=0)
                extra = dict(extra or {}, **dict(zip(AP_DIAGNOSTIC_FIELDS, map(float, values))))
            if finish_episode(ep, rows, extra):
                return None
    training_seconds = time.perf_counter() - t0
    save_agent(out / "model.pt", agent, cfg, name, n, seed)
    evaluation_started = time.perf_counter()
    summary = evaluate_agent(cfg, name, agent, n, seed, tr["eval_episodes"], details_dir=out)
    summary["final_lambda"] = float(getattr(agent, "lambda_val", 0.0))
    summary["final_alpha"] = float(getattr(agent, "alpha", 0.0))
    summary["train_seconds"] = training_seconds
    summary["evaluation_seconds"] = time.perf_counter() - evaluation_started
    summary["training_steps"] = tr["episodes"] * env.max_ep_steps + (tr["warmup_steps"] if replay is not None else 0)
    summary["mean_lambda_last_window"] = float(np.mean([row["lambda"] for row in history[-100:]]))
    summary["lambda_std_last_window"] = float(np.std([row["lambda"] for row in history[-100:]]))
    summary["cost_critic_architecture"] = cfg["algorithms"][name].get("cost_critic_architecture", "mlp")
    summary["reward_critic_architecture"] = cfg["algorithms"][name].get("reward_critic_architecture", "mlp")
    summary["actor_architecture"] = cfg["algorithms"][name].get("actor_architecture", "mlp")
    summary["parameters_actor"] = sum(p.numel() for p in agent.actor.parameters())
    summary["parameters_actor_trainable"] = sum(p.numel() for p in agent.actor.parameters() if p.requires_grad)
    summary["ap_entropy_mode"] = getattr(agent, "ap_entropy_mode", "legacy_sampled")
    summary["freeze_ap_policy"] = getattr(agent, "freeze_ap_policy", False)
    summary["matched_entropy_temperature"] = getattr(agent, "matched_entropy_temperature", False)
    summary["ap_gradient_mode"] = getattr(agent, "ap_gradient_mode", "straight_through")
    summary["layout_seed"] = int(cfg.get("layout_seed", cfg["layout_seed_base"]+seed))
    summary["train_seed"] = seed
    if summary["actor_architecture"] == "service_flow":
        summary["actor_hidden"] = cfg["algorithms"][name].get("actor_hidden", 64)
        summary["actor_embedding_dim"] = cfg["algorithms"][name].get("actor_embedding_dim", 8)
        for flag in ("physics_features", "pooled_context", "device_embedding"):
            summary["actor_use_"+flag] = cfg["algorithms"][name].get("actor_use_"+flag, True)
    if name == "safe":
        summary.update(training_cost_limit=agent.cost_limit, evaluation_cost_limit=cfg["cost_limit"],
                       penalty_mode=agent.penalty_mode, dual_cost_source=agent.dual_cost_source,
                       dual_risk_mode=agent.dual_risk_mode, excess_budget=agent.excess_budget,
                       actor_grad_clip=agent.actor_grad_clip, actor_update_interval=agent.actor_update_interval)
        summary.update(freeze_alpha=agent.freeze_alpha,
                       replay_recent_fraction=cfg["algorithms"][name].get("replay_recent_fraction", 0.),
                       replay_recent_window=cfg["algorithms"][name].get("replay_recent_window", 20000))
        if agent.dual_risk_mode == "excess":
            summary["excess_budget_met_in_eval"] = summary["mean_budget_exceedance_pp"] <= 100 * agent.excess_budget
    if cfg.get("intervention"):
        summary["shared_initial_episodes"] = cfg["intervention"]["parent_episode"]
        summary["additional_training_episodes"] = tr["episodes"] - cfg["intervention"]["parent_episode"]
    summary["actual_device"] = str(next(agent.actor.parameters()).device)
    if str(device).startswith("cuda"):
        summary["gpu_name"] = torch.cuda.get_device_name()
        summary["peak_gpu_memory_MB"] = torch.cuda.max_memory_allocated() / 1024**2
    # TD3 names its critics q1/q2/qc1/qc2; SAC uses critic_1/critic_2/cost_critic_1/cost_critic_2.
    reward_critics = (getattr(agent, "critic_1", None) or getattr(agent, "q1", None),
                      getattr(agent, "critic_2", None) or getattr(agent, "q2", None))
    cost_critic = getattr(agent, "cost_critic_1", None) or getattr(agent, "qc1", None)
    if getattr(agent, "structured_energy_enabled", False):
        summary["energy_hidden"] = cfg["algorithms"][name].get("energy_hidden", 64)
        summary["energy_component_loss_weight"] = agent.energy_component_loss_weight
        summary["parameters_per_reward_critic"] = sum(p.numel() for p in reward_critics[0].parameters())
        summary["learned_energy_factor_1"] = reward_critics[0].log_energy_factor.detach().clamp(-2.0, 2.0).exp().cpu().tolist()
        summary["learned_energy_factor_2"] = reward_critics[1].log_energy_factor.detach().clamp(-2.0, 2.0).exp().cpu().tolist()
    if getattr(agent, "structured_cost_enabled", False) or getattr(agent, "structured_energy_enabled", False):
        summary["cost_delivery_hidden"] = cfg["algorithms"][name].get("cost_delivery_hidden", 64)
        summary["cost_delivery_loss_weight"] = agent.cost_delivery_loss_weight
        summary["parameters_per_cost_critic"] = sum(p.numel() for p in cost_critic.parameters())
    return summary


def _log_episode(out, ep, rows, agent, t0, extra=None):
    arr = np.array(rows)
    row = {"episode": ep, "cost_percent": 100 * arr[:, 0].mean(), "energy_mJ_device_slot": 1000 * arr[:, 1].mean(),
           "avg_offload_ratio": arr[:, 2].mean(), "avg_tx_power": arr[:, 3].mean(),
           "lambda": float(getattr(agent, "lambda_val", 0.0)), "alpha": float(getattr(agent, "alpha", 0.0)),
           "elapsed_s": time.perf_counter() - t0}
    if hasattr(agent, "buf"):
        extra = extra or {}
        row.update(cpo_case=extra.get("cpo_case"), cpo_update_accepted=extra.get("accepted"),
                   cpo_rollout_cost=extra.get("J_c"))
    if getattr(agent, "structured_cost_enabled", False):
        row.update(extra or {"loss_qc": 0.0, "loss_delivery": 0.0})
    append_csv(out / "training.csv", row, ep == 1)
    write_json(out / "progress.json", row)
    if ep == 1 or ep % 10 == 0:
        print(f"ep {ep} cost={row['cost_percent']:.2f}% E={row['energy_mJ_device_slot']:.2f}mJ "
              f"lambda={row['lambda']:.3f} t={row['elapsed_s']:.0f}s", flush=True)
    return row


def run_lyapunov(cfg, n, seed, out):
    seed_all(seed)
    env = make_env(cfg, n, seed)
    agent = make_agent(cfg, "lyapunov", env)
    t0 = time.perf_counter()
    for ep in range(1, cfg["training"]["episodes"] + 1):
        state, done, rows = env.reset(), False, []
        while not done:
            action = execution_action(agent.select_action(state), env, "lyapunov")
            state, _, done, info = env.step(action)
            agent.update_virtual_queue(info["cost"], info)
            rows.append((info["cost"], info["energy"], info["executed_offload_ratio_mean"], info["tx_power_mean"]))
        if ep == 1 or ep % 25 == 0 or ep == cfg["training"]["episodes"]:
            _log_episode(out, ep, rows, agent, t0)
    summary = evaluate_agent(cfg, "lyapunov", agent, n, seed, cfg["training"]["eval_episodes"])
    summary.update(final_lambda=float(agent.lambda_val), train_seconds=time.perf_counter() - t0)
    return summary


def knob_grid(cfg, name):
    b = cfg["baselines"]
    if name == "const_ratio":
        return [{"x": float(x), "p": float(p)} for x in b["const_x_grid"] for p in b["const_p_grid"]]
    if name == "mb_greedy":
        return [{"slack": float(s)} for s in b["mb_slack_grid"]]
    return [{}]


def run_heuristic(cfg, name, n, seed, out, frontier=False):
    """Tune the knob on validation episodes (min energy s.t. cost <= d), then evaluate."""
    grid = knob_grid(cfg, name)
    vals = []
    for knob in grid:
        pol = make_heuristic(name, knob)
        s = evaluate_policy(cfg, n, seed, lambda env, st, p=pol: p.act(env),
                            cfg["baselines"]["validation_episodes"], val_seed(n, seed, cfg.get("validation_seed_base", 88000)))
        vals.append({**knob, "val_cost_percent": s["cost_percent"], "val_energy": s["energy_mJ_device_slot"]})
    write_json(out / "validation.json", {"grid": vals})
    d = 100 * cfg["cost_limit"]
    feas = [v for v in vals if v["val_cost_percent"] <= d]
    chosen = min(feas, key=lambda v: v["val_energy"]) if feas else min(vals, key=lambda v: v["val_cost_percent"])
    knob = {k: chosen[k] for k in grid[0]}
    pol = make_heuristic(name, knob)
    summary = evaluate_policy(cfg, n, seed, lambda env, st: pol.act(env), cfg["training"]["eval_episodes"],
                              eval_seed(n, seed, cfg.get("evaluation_seed_base", 99000)), details_dir=out)
    summary.update({f"knob_{k}": v for k, v in knob.items()})
    if frontier and len(grid) > 1:
        pts = []
        for k in grid:
            p = make_heuristic(name, k)
            s = evaluate_policy(cfg, n, seed, lambda env, st, p=p: p.act(env), cfg["training"]["eval_episodes"],
                                eval_seed(n, seed, cfg.get("evaluation_seed_base", 99000)))
            pts.append({**k, **{m: s[m] for m in ("cost_percent", "energy_mJ_device_slot", "bfsr_percent")}})
        write_json(out / "frontier.json", {"points": pts})
    return summary


def run_job(job_path):
    """Entry point for a worker process: job.json -> final_eval_summary.json."""
    job = read_json(job_path)
    out = Path(job_path).parent
    cfg = job["config"]
    from v2.plan import job_hash
    if job_hash(cfg, job["method"]) != job["config_hash"]:
        raise ValueError("Job source/configuration changed after planning")
    status = {"status": "running", "start": time.strftime("%Y-%m-%d %H:%M:%S"), "pid": os.getpid()}
    stopping = [False]
    def request_stop(signum, frame):
        stopping[0] = True
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)
    write_json(out / "run_status.json", status)
    write_json(out / "runtime.json", {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                                      "platform": platform.platform(), "cpu": platform.processor(),
                                      "device": cfg.get("execution", {}).get("device", "cpu"),
                                      "configured_torch_threads": cfg["training"]["torch_threads"],
                                      "cuda_version": torch.version.cuda, "source_digest": source_digest()})
    try:
        kind, name, n, seed = job["kind"], job["method"], job["N"], job["seed"]
        preview = make_env(cfg, n, seed)
        write_json(out / "environment.json", dict(parameters=preview.params, local_cpu=preview.local_cpu,
                   energy_factor=preview.energy_factor, ed_positions=preview.ed_pos, ap_positions=preview.ap_pos,
                   large_scale_gain_db=preview.large_scale_db, state_dim=preview.state_dim,
                   action_dim=preview.action_dim))
        if kind == "learner":
            summary = run_learner(cfg, name, n, seed, out, device=cfg.get("execution", {}).get("device", "cpu"),
                                  stop_requested=lambda: stopping[0])
        elif kind == "lyapunov":
            summary = run_lyapunov(cfg, n, seed, out)
        else:
            summary = run_heuristic(cfg, name, n, seed, out, frontier=job.get("frontier", False))
        if summary is None:
            status.update(status="interrupted", end=time.strftime("%Y-%m-%d %H:%M:%S"),
                          reason="Stop requested; episode-boundary checkpoint saved")
            write_json(out / "run_status.json", status)
            return
        write_json(out / "final_eval_summary.json", {"config_hash": job["config_hash"], "source_digest": source_digest(), "config": cfg, "job": {
            k: job[k] for k in ("experiment", "variant", "method", "N", "seed")}, "summary": summary})
        status.update(status="done", end=time.strftime("%Y-%m-%d %H:%M:%S"))
        if (out / "checkpoint.pt").exists() and not cfg["training"].get("keep_full_checkpoint", False):
            atomic_save(out / "checkpoint.pt", {"complete": True, "identity": job["config_hash"]})
    except BaseException:
        status.update(status="failed", error=traceback.format_exc())
        write_json(out / "run_status.json", status)
        raise
    write_json(out / "run_status.json", status)


if __name__ == "__main__":
    run_job(sys.argv[1])
