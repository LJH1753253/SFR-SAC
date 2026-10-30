import json

import pytest

from original_ch5.protocol import Catalog, EXPECTED_SOURCE, ROOT, STAGES, definition
from original_ch5.runner import complete
from v2.checkpoint import source_digest


@pytest.fixture(scope="module")
def catalog(tmp_path_factory):
    return Catalog(tmp_path_factory.mktemp("plan")).build()


def test_frozen_source():
    assert source_digest() == EXPECTED_SOURCE


def test_original_grid_sizes(catalog):
    assert len(catalog.cases) == 197
    assert len(catalog.parents) == 182
    assert len(catalog.tests) == 182
    assert {c["experiment"] for c in catalog.cases} == set(STAGES)


def test_no_new_beta_values(catalog):
    assert {c["beta"] for c in catalog.cases if c["experiment"] == "penalty"} == {1, 5, 10, 20}


def test_original_eta_and_budget(catalog):
    assert {c["eta"] for c in catalog.cases if c["experiment"] == "eta"} == {i/10 for i in range(1, 10)}
    assert {c["d"] for c in catalog.cases if c["experiment"] == "budget"} == {.08, .09, .10, .11, .12}


def test_default_aliases_are_shared(catalog):
    rows = [c for c in catalog.cases if c["N"] == 70 and c["seed"] == 0 and c["agent"] == "safe"
            and c["d"] == .10 and c["eta"] == .30 and c["b0"] == .05 and c["variant"] not in {"step", "episode"}]
    assert len(rows) == 5
    assert len({c["parent_id"] for c in rows}) == 1
    assert len({c["test_id"] for c in rows}) == 1


def test_replicate_definition(catalog):
    for c in catalog.cases:
        assert c["layout_seed"] == 1000+c["seed"]
        assert c["seed"] in (range(5) if c["experiment"] == "main" else range(3))


def test_no_ineffective_cost_target_arm(catalog):
    assert "cost_target" in definition()["blocked_experiments"]
    assert all(p["config"]["training"]["gamma"] == 0 for p in catalog.parents.values())
    assert all(p["config"]["algorithms"]["safe"]["cost_critic_target_mode"] == "immediate"
               for p in catalog.parents.values() if p["method"] == "safe")


def test_new_evaluation_stream(catalog):
    for test in catalog.tests.values():
        assert test["test_rng_seed"] == 2900000+100*test["N"]+test["seed"]
        assert test["test_episodes"] == 100


def test_same_nominal_lambda_learning_rate(catalog):
    for c in catalog.cases:
        if c["experiment"] == "dual":
            assert catalog.parents[c["parent_id"]]["config"]["algorithms"]["safe"]["lambda_lr"] == .005


def test_primary_actor_is_frozen(catalog):
    for c in catalog.cases:
        p = catalog.parents[c["parent_id"]]
        if c["agent"] == "safe":
            assert p["config"]["algorithms"]["safe"]["actor_architecture"] == "split_service_flow"
        assert p["config"]["environment"]["force_ap"] == "none"
        assert p["config"]["environment"]["energy_ref"] == .02


def test_shipped_reuse_counts(catalog):
    # The public repository intentionally omits the private cloud reuse cache.
    assert sum(p["reused"] for p in catalog.parents.values()) == 0
    assert sum(not p["reused"] for p in catalog.parents.values()) == 182
    main_ids = {c["parent_id"] for c in catalog.cases if c["experiment"] == "main"}
    assert sum(not catalog.parents[i]["reused"] for i in main_ids) == len(main_ids)


def test_reused_sources_not_scheduled_for_writes(catalog):
    for stage in STAGES:
        assert all(not t.get("reused", False) for t in catalog.tasks(stage))
        assert all(not t["directory"].startswith(str(ROOT / "reuse")) for t in catalog.tasks(stage))


def test_wrong_result_identity_is_rejected(tmp_path):
    spec = dict(task="test", directory=str(tmp_path), id="expected")
    (tmp_path / "test_summary.json").write_text(json.dumps(dict(source_digest="wrong")))
    with pytest.raises(RuntimeError):
        complete(spec)
