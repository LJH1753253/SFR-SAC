import hashlib
from pathlib import Path

from cloud24 import plan
from v2.checkpoint import source_digest
from v2.runtime import fingerprint, read_json


EXPECTED_SOURCE = "e942cebacf7dd4896d0a5b5647a47fb9c8246d7982c2fe21d72ba58b2f01d7d3"
ROOT = Path(__file__).resolve().parents[1]
STAGES = ("main", "budget", "eta", "dual", "penalty", "background")
METHODS = ("split_exact", "penalty_beta5", "td3", "dpp_rule", "vanilla")
BLOCKED_COST = (
    "Not scheduled: gamma=0 makes immediate and discounted_average targets identical. "
    "The frozen service-flow critics model immediate physical outcomes. A nonzero-discount "
    "comparison needs an explicitly specified compatible critic/control configuration; "
    "no such choice is silently introduced in this increment."
)


def addon_digest():
    result = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        result.update(path.name.encode())
        result.update(path.read_bytes())
    return result.hexdigest()


def definition(quick=False):
    return dict(
        version=1, frozen_source=EXPECTED_SOURCE, addon_digest=addon_digest(),
        selected_policy="split_exact", policy_selection=False, action_repair=False,
        main_N=[50, 55, 60, 65, 70], main_pairs=[[1000+i, i] for i in range(5)],
        ablation_N=70, ablation_pairs=[[1000+i, i] for i in range(3)],
        algorithms=list(METHODS), budget=[.08, .09, .10, .11, .12],
        eta=[i/10 for i in range(1, 10)], dual=["batch", "episode", "step"],
        penalty_beta=[1, 5, 10, 20], background=[0., .05, .10],
        training_episodes=2 if quick else 1600, horizon=12 if quick else 200,
        final_evaluation_episodes=1 if quick else 100,
        training_validation_episodes=1 if quick else 30,
        validation_note="Retain legacy training configuration identity; all reported final tests use 100 episodes.",
        default_eta=.30, default_d=.10, default_lambda_lr=.005,
        gamma=0., energy_ref_J=.02,
        energy_unit="mJ/device/slot", cost_unit="percent",
        evaluation_stream="2900000 + 100*N + layout_seed - 1000",
        training_seed_layout_note="Five paired deployment/training replicates, not five seeds per deployment.",
        dual_note="Only granularity changes. The same lambda_lr is used: 40 batch, 200 step, or 1 episode update per episode; this is not an equal-effective-gain comparison.",
        baseline_note="Penalty/Vanilla retain frozen shared service-flow actors; actor-matched split penalties are not substituted. Lyapunov is a heuristic with its frozen 20-episode virtual-queue warm-up, not an exact DPP optimizer.",
        original_grid_source="paper1_clean_release_20261014/configs/paper.json: suite",
        blocked_experiments=dict(cost_target=BLOCKED_COST),
        not_included=["candidate reselection", "new beta values", "automatic default selection",
                      "CPO/reward-horizon/hinge/AP/latency revision extensions"],
        quick=quick,
    )


class Catalog:
    def __init__(self, output, quick=False):
        if source_digest() != EXPECTED_SOURCE:
            raise RuntimeError("Frozen core source/configuration digest mismatch")
        self.output = Path(output).resolve()
        self.quick = quick
        self.parents = {}
        self.tests = {}
        self.cases = []
        index = ROOT / "reuse" / "index.json"
        self.index = read_json(index)["models"] if index.exists() and not quick else {}

    def parent(self, variant, n, seed, change=None):
        selected = "shared_exact" if variant.startswith("penalty") or variant == "vanilla" else "split_exact"
        cfg = plan.configure(variant, n=n, layout=1000+seed, train_seed=seed,
                             selected=selected, small=self.quick)
        if change:
            change(cfg)
        method = cfg["method"]
        identity = plan.job_hash(cfg, method)
        if identity in self.parents:
            return self.parents[identity]
        directory = self.output / "runs" / f"N{n}" / f"layout{1000+seed}" / f"train{seed}" / identity
        spec = dict(id=identity, task="train", kind="learner" if method in
                    {"safe", "vanilla", "penalty_sac", "td3_lag"} else "lyapunov",
                    method=method, N=n, seed=seed, layout_seed=1000+seed,
                    variant=variant, experiment="original_ch5", config=cfg, config_hash=identity,
                    directory=str(directory), resource="gpu" if cfg["execution"]["require_cuda"] else "cpu",
                    reused=False)
        if identity in self.index:
            asset = self.index[identity]
            parent = ROOT / asset["directory"]
            saved = read_json(parent / "job.json")
            final = read_json(parent / "final_eval_summary.json")
            if plan.job_hash(saved["config"], saved["method"]) != identity or final["config_hash"] != identity:
                raise RuntimeError(f"Reused configuration mismatch: {parent}")
            if final["source_digest"] != EXPECTED_SOURCE:
                raise RuntimeError(f"Reused model has the wrong source: {parent}")
            if saved["kind"] == "learner" and not (parent / "model.pt").exists():
                raise RuntimeError(f"Reused trained model is missing: {parent}")
            spec.update(config=saved["config"], directory=str(parent), reused=True,
                        reused_origin=asset["origin"])
        self.parents[identity] = spec
        return spec

    def evaluation(self, parent):
        seed = 2900000 + 100*parent["N"] + parent["layout_seed"]-1000
        identity = fingerprint(dict(parent_id=parent["id"], rng=seed,
                                   episodes=1 if self.quick else 100,
                                   source=EXPECTED_SOURCE, addon=addon_digest()))
        if identity not in self.tests:
            spec = plan.test_job(self.output, parent, parent["variant"], small=self.quick)
            spec.update(id=identity, parent_id=parent["id"], test_rng_seed=seed,
                        test_episodes=1 if self.quick else 100, split="original_ch5_fixed_test",
                        addon_digest=addon_digest(), directory=str(self.output / "test" / identity))
            self.tests[identity] = spec
        return self.tests[identity]

    def add(self, stage, label, parent):
        evaluation = self.evaluation(parent)
        cfg = parent["config"]
        self.cases.append(dict(experiment=stage, variant=label, agent=parent["method"],
                               N=parent["N"], seed=parent["seed"], layout_seed=parent["layout_seed"],
                               eta=cfg["environment"]["ul_time_fraction"], d=cfg["cost_limit"],
                               b0=cfg["environment"]["mec_background_load_base"],
                               beta=cfg["algorithms"][parent["method"]].get("penalty_sac_beta"),
                               parent_id=parent["id"], test_id=evaluation["id"]))

    def build(self):
        for seed in range(5):
            for n in (50, 55, 60, 65, 70):
                for variant in METHODS:
                    self.add("main", variant, self.parent(variant, n, seed))
        for seed in range(3):
            for d in (.08, .09, .10, .11, .12):
                self.add("budget", f"d{d:.2f}", self.parent("split_exact", 70, seed,
                         lambda c, d=d: c.update(cost_limit=d)))
            for eta in (i/10 for i in range(1, 10)):
                self.add("eta", f"eta{eta:.2f}", self.parent("split_exact", 70, seed,
                         lambda c, eta=eta: c["environment"].update(ul_time_fraction=eta)))
            for mode in ("batch", "episode", "step"):
                self.add("dual", mode, self.parent("split_exact", 70, seed,
                         lambda c, mode=mode: c["algorithms"]["safe"].update(lambda_update_granularity=mode)))
            for beta in (1, 5, 10, 20):
                self.add("penalty", f"beta{beta}", self.parent(f"penalty_beta{beta}", 70, seed))
            for b0 in (0., .05, .10):
                self.add("background", f"b0_{b0:.2f}", self.parent("split_exact", 70, seed,
                         lambda c, b0=b0: c["environment"].update(mec_background_load_base=b0)))
        return self

    def tasks(self, stage):
        cases = [c for c in self.cases if c["experiment"] == stage]
        tasks = {}
        for case in cases:
            parent = self.parents[case["parent_id"]]
            if not parent["reused"]:
                tasks[parent["id"]] = parent
            test = self.tests[case["test_id"]]
            tasks[test["id"]] = test
        return list(tasks.values())

    def manifest(self):
        return dict(protocol=definition(self.quick), cases=self.cases,
                    parents=list(self.parents.values()), evaluations=list(self.tests.values()))
