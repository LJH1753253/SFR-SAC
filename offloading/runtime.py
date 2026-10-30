"""Configuration, policy construction, and deterministic evaluation."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from agents.action_projection import a_ap_to_rsu_index, project_action_numpy
from agents.lyapunov_dpp_agent import LyapunovDPPAgent
from agents.safe_sac_agent import SafeSACAgent, VanillaSACAgent
from agents.td3_agent import TD3Agent, TD3Config
from envs.vec_offloading_env import VecOffloadingEnv


ROOT = Path(__file__).resolve().parents[1]
AGENTS = ("safe", "penalty_sac", "td3_lag", "lyapunov", "vanilla")
MODULES = (
    "actor",
    "actor_target",
    "critic_1",
    "critic_2",
    "critic_1_target",
    "critic_2_target",
    "cost_critic_1",
    "cost_critic_2",
    "cost_critic_1_target",
    "cost_critic_2_target",
    "q1",
    "q2",
    "q1_target",
    "q2_target",
    "qc1",
    "qc2",
    "qc1_target",
    "qc2_target",
)


def read_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "task_model": "oneshot",
        "action_mode": "hybrid_ap_gumbel",
        "tx_power_control": "action",
        "local_energy_model": "physical_joule_cpu",
        "tx_energy_model": "physical_pa_circuit",
        "reward_mode": "energy",
        "oneshot_cost_components": "full",
    }
    for key, expected in required.items():
        if config["environment"].get(key) != expected:
            raise ValueError(f"This release requires {key}={expected}")
    if config["tx_energy_accounting"] not in {"delivered", "attempted"}:
        raise ValueError("tx_energy_accounting must be delivered or attempted")
    return config


def fingerprint(config: dict) -> str:
    raw = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_name(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def make_env(config: dict, n: int) -> VecOffloadingEnv:
    kwargs = dict(config["environment"])
    kwargs.update(
        num_devices=n,
        max_episodes=config["training"]["episodes"],
        tx_energy_accounting=config["tx_energy_accounting"],
    )
    return VecOffloadingEnv(**kwargs)


def make_agent(config: dict, name: str, env: VecOffloadingEnv, device: str):
    cfg = config["algorithms"][name]
    common = dict(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        device=device,
        max_action=1.0,
        num_devices=env.N,
        num_rsu=env.R,
        ap_projection_mode="hybrid_ap_gumbel",
    )
    dual = (
        dict(
            cost_limit=config.get("cost_limit", 0.10),
            lambda_lr=cfg["lambda_lr"],
            lambda_mode=cfg["lambda_mode"],
            fixed_lambda=cfg["fixed_lambda"],
            initial_lambda=cfg["initial_lambda"],
            slow_lambda_factor=cfg["slow_lambda_factor"],
            cost_coef=cfg["cost_coef"],
            update_lambda_in_batch=cfg["lambda_update_granularity"] == "batch",
            cost_critic_target_mode=cfg["cost_critic_target_mode"],
        )
        if name in {"safe", "td3_lag"}
        else {}
    )
    if name == "td3_lag":
        td3 = TD3Config(
            lr=cfg["td3_lr"],
            tau=cfg["td3_tau"],
            policy_noise=cfg["td3_policy_noise"],
            exploration_noise=cfg["td3_exploration_noise"],
            exploration_noise_rel_scale=cfg["td3_exploration_noise_rel_scale"],
            exploration_noise_ap_scale=cfg["td3_exploration_noise_ap_scale"],
            policy_noise_rel_scale=cfg["td3_policy_noise_rel_scale"],
            policy_noise_ap_scale=cfg["td3_policy_noise_ap_scale"],
        )
        td3.gamma = config["training"]["gamma"]
        return TD3Agent(**common, **dual, cfg=td3, use_lagrangian=True)
    if name == "lyapunov":
        keys = {
            "load_scale": "load_scale",
            "slot_duration": "slot_duration",
            "ul_time_fraction": "ul_time_fraction",
            "total_access_bw": "total_access_bw",
            "bs_cpu_base": "bs_cpu_base",
            "max_buffer": "max_buffer",
            "tx_power": "tx_power_base",
            "tx_power_control": "tx_power_control",
            "tx_power_min": "tx_power_min",
            "tx_power_max": "tx_power_max",
            "local_energy_per_bit": "local_energy_per_bit",
            "tx_energy_per_bit": "tx_energy_per_bit",
            "task_model": "task_model",
        }
        kwargs = {key: config["environment"][value] for key, value in keys.items()}
        options = (
            "mode",
            "queue_target",
            "max_offload",
            "access_margin",
            "edge_margin",
            "v_energy",
            "eta_delivery",
            "eta_queue",
            "delivery_limit",
            "queue_limit",
            "initial_z",
        )
        kwargs.update({key: cfg[f"lyapunov_{key}"] for key in options})
        kwargs["delivery_limit"] = config.get("cost_limit", 0.10)
        return LyapunovDPPAgent(num_devices=env.N, num_rsu=env.R, **kwargs)
    common.update(
        gamma=config["training"]["gamma"],
        alpha=cfg["fixed_alpha"],
        auto_alpha=bool(cfg["auto_alpha"]),
        alpha_lr=cfg["alpha_lr"],
        target_entropy=-float(env.N),
        action_mode="hybrid_ap_gumbel",
        gumbel_tau=cfg["gumbel_tau"],
        gumbel_tau_min=cfg["gumbel_tau_min"],
        ap_entropy_weight=cfg["ap_entropy_weight"],
    )
    if name == "safe":
        return SafeSACAgent(**common, **dual, lr=cfg["safe_lr"], tau=cfg["safe_tau"])
    return VanillaSACAgent(**common, lr=cfg["vanilla_lr"], tau=cfg["vanilla_tau"])


def execution_action(action, env: VecOffloadingEnv, name: str):
    arr = np.asarray(action, dtype=np.float32)
    if name == "lyapunov":
        n = env.N
        indices = a_ap_to_rsu_index(arr[2 * n : 3 * n], env.R, mode="legacy")
        onehot = np.eye(env.R, dtype=np.float32)[indices]
        arr = np.concatenate([arr[:n], arr[3 * n : 4 * n], onehot.ravel()])
    return project_action_numpy(arr, num_devices=env.N, num_rsu=env.R, mode="hybrid_ap_gumbel")


def training_reward(reward: float, cost: float, config: dict, name: str) -> float:
    value = config["training"]["reward_scale"] * reward
    if name == "penalty_sac":
        value -= config["algorithms"][name]["penalty_sac_beta"] * cost
    return float(value)


def slot_metrics(info: dict, n: int, budget: float) -> dict:
    cost = float(info["service_failure_ratio"])
    return {
        "cost_percent": 100.0 * cost,
        "energy_mJ_device_slot": 1000.0 * float(info["energy"]),
        "bfsr_percent": 100.0 * float(cost <= budget),
        "local_failure_percent": 100.0 * float(info["local_failure_ratio"]),
        "edge_delivery_failure_percent": 100.0 * float(info["edge_arrival_failure_ratio"]),
        "mec_service_failure_percent": 100.0 * float(info["mec_admission_failure_ratio"]),
        "E_loc_J_device_slot": float(info["local_energy"]) / n,
        "E_tx_J_device_slot": float(info["tx_energy"]) / n,
        "avg_tx_power": float(info["tx_power_mean"]),
        "avg_offload_ratio": float(info["executed_offload_ratio_mean"]),
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(agent, config: dict, name: str, n: int, seed: int, episodes: int):
    seed_all(seed)
    env = make_env(config, n)
    rows = []
    slot_costs = []
    for episode in range(1, episodes + 1):
        state = env.reset()
        values = []
        while True:
            action = execution_action(agent.select_action(state, deterministic=True), env, name)
            state, _, done, info = env.step(action)
            values.append(slot_metrics(info, n, config.get("cost_limit", 0.10)))
            slot_costs.append(float(info["service_failure_ratio"]))
            if done:
                break
        rows.append(
            {
                "episode": episode,
                **{key: float(np.mean([r[key] for r in values])) for key in values[0]},
            }
        )
    summary = {key: float(np.mean([r[key] for r in rows])) for key in rows[0] if key != "episode"}
    summary["E_tx_over_E_loc"] = summary["E_tx_J_device_slot"] / max(
        summary["E_loc_J_device_slot"], 1e-12
    )
    summary["cost_p95_percent"] = float(100.0 * np.percentile(slot_costs, 95))
    summary["eval_episodes"] = episodes
    summary["eval_slots"] = len(slot_costs)
    summary["final_lambda"] = float(getattr(agent, "lambda_val", 0.0))
    return summary, rows


def save_model(path: Path, agent, config: dict, name: str, n: int, seed: int):
    modules = {key: getattr(agent, key).state_dict() for key in MODULES if hasattr(agent, key)}
    scalars = {
        key: float(getattr(agent, key)) for key in ["alpha", "lambda_val"] if hasattr(agent, key)
    }
    state = {
        "agent": name,
        "N": n,
        "seed": seed,
        "config": config,
        "modules": modules,
        "scalars": scalars,
    }
    if modules:
        torch.save(state, path)
        torch.save({**state, "modules": {"actor": modules["actor"]}}, path.with_name("actor.pt"))
    else:
        state["scalars"].update(z_delivery=agent.z_delivery, z_queue=agent.z_queue)
        write_json(path.with_suffix(".json"), state)


def load_model(path: Path, config: dict, device: str):
    checkpoint = (
        json.loads(path.read_text())
        if path.suffix == ".json"
        else torch.load(path, map_location=device, weights_only=True)
    )
    name = checkpoint["agent"]
    n = int(checkpoint.get("N", checkpoint["config"].get("num_devices", 70)))
    env = make_env(config, n)
    agent = make_agent(config, name, env, device)
    modules = checkpoint.get(
        "modules", {"actor": checkpoint["actor"]} if "actor" in checkpoint else {}
    )
    for key, state in modules.items():
        getattr(agent, key).load_state_dict(state, strict=True)
    for key in ["alpha", "lambda_val"]:
        if key in checkpoint.get("scalars", {}):
            setattr(agent, key, float(checkpoint["scalars"][key]))
    if name == "lyapunov":
        for key in ["z_delivery", "z_queue"]:
            setattr(agent, key, float(checkpoint["scalars"][key]))
    return agent, name, n


def resolved_config(config: dict, name: str, n: int, seed: int) -> dict:
    result = copy.deepcopy(config)
    result.update(agent=name, N=n, seed=seed, cost_limit=config.get("cost_limit", 0.10))
    result["state_dim"] = 3 * n + result["environment"]["num_rsu"] + 8
    result["action_dim"] = 2 * n + n * result["environment"]["num_rsu"]
    return result
