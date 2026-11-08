"""Fast checks of the revision extensions (CPU, tiny configurations)."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from offloading.runtime import execution_action
from revision import variants as V
from revision.per_slot import CriticOptimizer, ExactSlotModel, OracleOptimizer
from revision.protocol import Catalog, STAGES
from v2.env import water_fill
from v2.runtime import load_agent, make_agent, make_env, run_learner


def small(variant, n=6):
    return V.configure(variant, n=n, small=True)


def test_water_fill_is_max_min_fair():
    assert np.allclose(water_fill([1, 2, 10, 20], 15), [1, 2, 6, 6])
    assert np.allclose(water_fill([1, 2, 3], 100), [1, 2, 3])


def test_max_min_sharing_only_changes_overloaded_slots():
    cfg = small("sfr", n=8)
    envs = []
    for sharing in ("proportional", "max_min_fair"):
        env = make_env(dict(cfg, environment=dict(cfg["environment"], mec_sharing=sharing)), 8, 0)
        np.random.seed(0)
        env.reset()
        envs.append(env)
    x, p, ap = np.ones(8), np.full(8, .2), np.arange(8) % envs[0].R
    a, b = (env.evaluate_slot(x, p, ap) for env in envs)
    assert np.allclose(a["y_mec"], b["y_mec"])  # enough MEC capacity: identical
    for env in envs:
        env.bs_cpu_nominal = 1e8  # force an overloaded MEC
    a, b = (env.evaluate_slot(x, p, ap) for env in envs)
    cycles = lambda out, env: (out["y_mec"] * env.task_cpb).sum()  # noqa: E731
    assert np.isclose(cycles(a, envs[0]), cycles(b, envs[1]))  # both use the whole capacity
    served = lambda out, env: (out["y_mec"] * env.task_cpb)[out["y_del"] > 0]  # noqa: E731
    assert served(b, envs[1]).min() >= served(a, envs[0]).min() - 1e-6  # max-min raises the smallest share


def test_exact_slot_model_matches_simulator():
    cfg = small("sfr", n=10)
    env = make_env(cfg, 10, 0)
    np.random.seed(1)
    env.reset()
    model = ExactSlotModel(env)
    tensors = model.tensors()
    for _ in range(5):
        x = np.random.uniform(0, 1, 10)
        x[np.random.rand(10) < .3] = 0
        p = np.random.uniform(env.tx_power_min, env.tx_power_max, 10)
        ap = np.random.randint(0, env.R, 10)
        out = env.evaluate_slot(x, p, ap)
        energy, cost = model.outcome(tensors, torch.as_tensor(x)[None], torch.as_tensor(p)[None],
                                     F.one_hot(torch.as_tensor(ap, dtype=torch.long), env.R).double()[None])
        assert np.isclose(float(energy[0]), out["energy"], rtol=1e-6)
        assert np.isclose(float(cost[0]), out["cost"], rtol=1e-6, atol=1e-9)


def test_model_assumptions_reach_structured_networks():
    cfg = small("mm_mec_capacity_sfr_wrong")
    env = make_env(cfg, 6, 0)
    agent = make_agent(cfg, "safe", env)
    assumed = V.MISMATCH["mec_capacity"]["assumptions"]["bs_cpu_base"]
    assert env.params["bs_cpu_base"] != assumed
    for module in (agent.actor, agent.critic_1, agent.cost_critic_1):
        assert module.p["bs_cpu_base"] == assumed


def test_max_gain_actor_executes_best_channel_ap():
    cfg = small("sfr_maxgain", n=8)
    env = make_env(cfg, 8, 0)
    agent = make_agent(cfg, "safe", env)
    np.random.seed(0)
    state = env.reset()
    for _ in range(10):
        action = execution_action(agent.select_action(state, deterministic=True), env, "safe")
        _, _, ap = env.decode_action(action)
        assert np.mean(ap == np.argmax(env._channel_db(), 1)) == 1.0
        state, _, _, _ = env.step(action)


def test_structured_td3_uses_sfr_networks():
    cfg = small("td3_struct_split")
    agent = make_agent(cfg, "td3_lag", make_env(cfg, 6, 0))
    reference = make_agent(small("sfr"), "safe", make_env(small("sfr"), 6, 0))
    assert type(agent.q1).__name__ == type(reference.critic_1).__name__
    assert type(agent.qc1).__name__ == type(reference.cost_critic_1).__name__
    assert type(agent.actor.policy).__name__ == type(reference.actor).__name__
    assert agent.structured_energy_enabled and agent.structured_cost_enabled


@pytest.mark.parametrize("variant", ["td3_struct", "td3_struct_split", "pen_struct_b3", "cpo_g0.99"])
def test_new_arms_train_and_reload(tmp_path, variant):
    cfg = small(variant)
    method = cfg["method"]
    summary = run_learner(cfg, method, 6, 0, tmp_path, device="cpu")
    assert np.isfinite(summary["cost_percent"]) and np.isfinite(summary["energy_mJ_device_slot"])
    agent, name, n, _ = load_agent(tmp_path / "model.pt", cfg)
    state = make_env(cfg, n, 0).reset()
    assert agent.select_action(state, deterministic=True).shape == (2 * n + n * 6,)


def test_fixed_penalty_arm_keeps_its_multiplier():
    cfg = small("pen_struct_b3")
    agent = make_agent(cfg, "safe", make_env(cfg, 6, 0))
    assert agent.lambda_mode == "fixed" and agent.penalty_mode == "linear"
    assert np.isclose(agent.lambda_val * agent.cost_coef, 3.0)


def test_per_slot_optimizers_return_valid_actions():
    cfg = small("sfr", n=6)
    env = make_env(cfg, 6, 0)
    agent = make_agent(cfg, "safe", env)
    np.random.seed(0)
    state = env.reset()
    for optimizer in (CriticOptimizer(agent, 3, start="actor"), CriticOptimizer(agent, 3, start="neutral"),
                      OracleOptimizer(cfg["cost_limit"], 3)):
        action = optimizer.act(env, state)
        x, p, ap = env.decode_action(action)
        assert x.shape == (6,) and np.all((0 <= x) & (x <= 1)) and np.all(ap < env.R)


def test_catalog_shares_identical_trainings(tmp_path):
    catalog = Catalog(tmp_path / "revision_catalog_test", quick=True).build()
    assert set(c["experiment"] for c in catalog.cases) == set(STAGES)
    sfr = [p for p in catalog.parents.values() if p["variant"] == "sfr"]
    assert len(sfr) == 1 and "sfr_hinge_d0.1" in sfr[0].get("aliases", [])
    td3 = [p for p in catalog.parents.values() if p["variant"] == "td3"]
    assert len(td3) == 1 and "td3_g0" in td3[0].get("aliases", [])
