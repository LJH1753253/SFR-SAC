"""Planned training runs, held-out tests, per-slot optimizer runs and the latency benchmark.

Pairs are the main study's paired deployments (layout 1000+i, training seed i). Final tests use the
same held-out stream formula as `original_ch5` (2900000 + 100 N + layout - 1000), so every arm in a
stage is evaluated on identical test episodes. Training identities include the source digest, and
identical configurations are trained once and shared between stages.
"""
from __future__ import annotations

from pathlib import Path

from cloud24 import plan
from revision import variants as V
from revision.per_slot import OPTIMIZERS
from revision.source import revision_source_digest
from v2.checkpoint import source_digest
from v2.plan import job_hash
from v2.runtime import fingerprint

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("fair_critics", "per_slot", "mismatch", "cpo", "horizon", "penalty_form", "ap", "latency")
LEARNERS = {"safe", "td3_lag", "cpo", "penalty_sac", "vanilla"}
MAIN_N = (50, 55, 60, 65, 70)


def test_seed(n, layout):
    return 2900000 + 100 * n + layout - 1000


def definition(quick=False):
    return dict(
        version=1, revision_source_digest=revision_source_digest(), stages=list(STAGES),
        pairs5=[list(p) for p in V.PAIRS5], pairs3=[list(p) for p in V.PAIRS3],
        test_stream="2900000 + 100*N + layout_seed - 1000", test_episodes=1 if quick else 100,
        per_slot_episodes=1 if quick else 30, per_slot_warmup_episodes=1 if quick else 10,
        per_slot_iterations=dict(critic_refine=3 if quick else 30, critic_scratch=3 if quick else 150,
                                 oracle=3 if quick else 100),
        penalty_betas=list(V.PENALTY_BETAS), mismatch={k: v for k, v in V.MISMATCH.items()},
        latency_sizes=[4, 6] if quick else [50, 70, 200], quick=quick,
        notes=dict(
            fair_critics="Safe TD3 and a fixed-penalty SAC receive SFR-SAC's structured critics (and, for "
                         "td3_struct_split, its split actor network), isolating the critic structure from the "
                         "constrained-RL algorithm.",
            per_slot="Direct per-slot optimization of the trained critics' objective and a full-information "
                     "oracle; tests whether the actor is needed beyond amortization (latency).",
            mismatch="The true environment differs from the physical constants given to SFR-SAC's structured "
                     "networks; model-free Safe TD3 and full-information MB-greedy are trained/tuned in the "
                     "same perturbed environment.",
            horizon="E1: gamma 0 vs 0.99 for learners with generic (MLP) critics; both arms use sampled AP "
                    "entropy because the exact entropy controls require gamma 0. CPO arms are in stage cpo.",
            cpo="E7: CPO with the split service-flow actor, gamma 0 and gamma 0.99, main grid N and pairs.",
            penalty_form="E2: hinge vs untruncated Lagrangian actor penalty for SFR-SAC at d = 8, 10, 12 %.",
            ap="E3: learned AP vs best-channel AP, and AP-dependent vs AP-independent gains.",
            latency="E6: one CPU thread; initialized networks for N = 50, 70, 200 plus the trained N = 70 actor."))


class Catalog:
    def __init__(self, output, quick=False):
        self.output = Path(output).resolve()
        self.quick = quick
        self.parents, self.tasks_by_id, self.cases = {}, {}, []
        self.definition = definition(quick)

    # ------------------------------------------------------------------ specs
    def parent(self, variant, n, layout, seed):
        cfg = V.configure(variant, n=n, layout=layout, seed=seed, small=self.quick)
        method = cfg["method"]
        identity = job_hash(cfg, method)
        if identity in self.parents:
            self.parents[identity].setdefault("aliases", []).append(variant)
            return self.parents[identity]
        directory = self.output / "runs" / f"N{n}" / f"layout{layout}" / f"train{seed}" / identity
        kind = "learner" if method in LEARNERS else "lyapunov" if method == "lyapunov" else "heuristic"
        spec = dict(id=identity, task="train", kind=kind, method=method, N=n, seed=seed, layout_seed=layout,
                    variant=variant, experiment="revision", config=cfg, config_hash=identity,
                    directory=str(directory), resource="gpu" if cfg["execution"].get("require_cuda") else "cpu",
                    revision_source_digest=revision_source_digest())
        self.parents[identity] = spec
        return spec

    def test(self, parent):
        seed = test_seed(parent["N"], parent["layout_seed"])
        episodes = self.definition["test_episodes"]
        identity = fingerprint(dict(parent_id=parent["id"], rng=seed, episodes=episodes,
                                    source=source_digest(), revision_source=revision_source_digest()))
        if identity not in self.tasks_by_id:
            spec = plan.test_job(self.output, parent, parent["variant"], small=self.quick)
            spec.update(id=identity, task="test", parent_id=parent["id"], test_rng_seed=seed, test_episodes=episodes,
                        revision_source_digest=revision_source_digest(),
                        directory=str(self.output / "test" / identity))
            self.tasks_by_id[identity] = spec
        return self.tasks_by_id[identity]

    def per_slot(self, parent, optimizer):
        d = self.definition
        n, layout = parent["N"], parent["layout_seed"]
        spec = dict(task="per_slot", optimizer=optimizer, parent=parent["directory"], parent_id=parent["id"],
                    N=n, seed=parent["seed"], layout_seed=layout, variant=f"per_slot_{optimizer}",
                    method="safe", resource="cpu", test_rng_seed=test_seed(n, layout),
                    test_episodes=d["per_slot_episodes"], iters=d["per_slot_iterations"].get(optimizer, 0),
                    warmup_episodes=d["per_slot_warmup_episodes"], warmup_rng_seed=2950000 + 100 * n + layout - 1000)
        spec["revision_source_digest"] = revision_source_digest()
        spec["id"] = fingerprint(dict(spec, source=source_digest()))
        spec["directory"] = str(self.output / "per_slot" / spec["id"])
        self.tasks_by_id.setdefault(spec["id"], spec)
        return self.tasks_by_id[spec["id"]]

    def latency(self, trained_parent):
        d = self.definition
        spec = dict(task="latency", parent=trained_parent["directory"], parent_id=trained_parent["id"],
                    trained_parent=trained_parent["directory"], N=70, seed=0, layout_seed=1000,
                    variant="latency", method="safe", resource="cpu", sizes=d["latency_sizes"],
                    warmup=5 if self.quick else 300, reps=20 if self.quick else 3000, small=self.quick)
        spec["revision_source_digest"] = revision_source_digest()
        spec["id"] = fingerprint(dict(spec, source=source_digest()))
        spec["directory"] = str(self.output / "latency" / spec["id"])
        self.tasks_by_id.setdefault(spec["id"], spec)
        return self.tasks_by_id[spec["id"]]

    # ------------------------------------------------------------------ cases
    def add(self, stage, label, parent, task=None):
        task = task or self.test(parent)
        self.cases.append(dict(experiment=stage, variant=label, method=parent["method"], N=parent["N"],
                               seed=parent["seed"], layout_seed=parent["layout_seed"], parent_id=parent["id"],
                               task_id=task["id"], task=task["task"]))

    def build(self):
        n70 = 6 if self.quick else 70
        pairs5, pairs3 = V.PAIRS5, V.PAIRS3
        if self.quick:
            pairs5 = pairs3 = V.PAIRS5[:1]
        ns = (4, 6) if self.quick else MAIN_N
        for layout, seed in pairs5:
            for variant in ("sfr", "td3", "td3_struct", "td3_struct_split"):
                self.add("fair_critics", variant, self.parent(variant, n70, layout, seed))
            sfr = self.parent("sfr", n70, layout, seed)
            for optimizer in OPTIMIZERS:
                self.add("per_slot", optimizer, sfr, self.per_slot(sfr, optimizer))
            for n in ns:
                for variant in ("cpo_g0", "cpo_g0.99"):
                    self.add("cpo", variant, self.parent(variant, n, layout, seed))
        for layout, seed in pairs3:
            for beta in V.PENALTY_BETAS:
                variant = f"pen_struct_b{beta:g}"
                self.add("fair_critics", variant, self.parent(variant, n70, layout, seed))
            for variant in ("sfr", "td3", "mb_greedy"):
                self.add("mismatch", f"nominal_{variant}", self.parent(variant, n70, layout, seed))
            for case in V.MISMATCH:
                for arm in V.MISMATCH_ARMS:
                    if arm == "sfr_right" and not V.MISMATCH[case]["assumptions"]:
                        continue
                    variant = f"mm_{case}_{arm}"
                    self.add("mismatch", variant, self.parent(variant, n70, layout, seed))
            for variant in ("sacmlp_g0", "sacmlp_g0.99", "td3_g0", "td3_g0.99", "pen5_g0", "pen5_g0.99"):
                self.add("horizon", variant, self.parent(variant, n70, layout, seed))
            for variant in ("sfr_hinge_d0.08", "sfr_hinge_d0.1", "sfr_hinge_d0.12",
                            "sfr_linear_d0.08", "sfr_linear_d0.1", "sfr_linear_d0.12"):
                self.add("penalty_form", variant, self.parent(variant, n70, layout, seed))
            for variant in ("sfr", "sfr_maxgain", "sfr_apind", "td3", "td3_apind", "mb_greedy", "mb_greedy_apind"):
                self.add("ap", variant, self.parent(variant, n70, layout, seed))
        first = self.parent("sfr", n70, *V.PAIRS5[0])
        self.add("latency", "latency", first, self.latency(first))
        return self

    def tasks(self, stage):
        tasks = {}
        for case in self.cases:
            if case["experiment"] != stage:
                continue
            parent = self.parents[case["parent_id"]]
            tasks[parent["id"]] = parent
            tasks[case["task_id"]] = self.tasks_by_id[case["task_id"]]
        return list(tasks.values())

    def manifest(self):
        return dict(protocol=self.definition, source_digest=source_digest(),
                    revision_source_digest=revision_source_digest(), cases=self.cases,
                    parents=list(self.parents.values()), tasks=list(self.tasks_by_id.values()))
