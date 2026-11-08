# Revision extensions (branch `revision-extensions`)

This branch adds the experiments the IoT-J reviewers asked for that the main study does not contain,
and two controls needed to interpret SFR-SAC's advantage. The main study (`original_ch5`,
`results/final`) is left as it is.

## Why these experiments

SFR-SAC's structured critics contain the simulator's slot model (local capacity and energy, MEC
capacity and proportional sharing, TDMA airtime energy); only the delivery success and the per-device
energy factors are learned. The main-grid baselines use generic MLP critics, and the mechanism
ablation shows that the structured critics account for most of the gain (split actor with MLP
critics: 11.45 % / 10.24 mJ against 9.38 % / 8.59 mJ). The paper therefore has to show
(1) what the constrained SAC adds once every learner has the same critics, (2) what a learned actor
adds over directly optimizing the same model in each slot, and (3) how the method behaves when its
assumed physics is wrong.

| Stage | Question (reviewer) | Arms | Pairs |
|---|---|---|---|
| `fair_critics` | Is the gain the critics or the algorithm? | SFR-SAC; Safe TD3 (MLP); Safe TD3 + SFR critics; Safe TD3 + SFR critics + SFR split actor; fixed penalty β ∈ {2, 2.5, 3, 3.5, 4.5} with SFR critics and actor | 5 (β sweep 3) |
| `per_slot` | Why RL? (R1.1) | trained actor; descent on the trained critics from the actor's action (30 steps) and from a neutral start (150 steps); full-information primal-dual oracle on the exact slot model (100 steps, multi-start incl. MB-greedy) — 30 test episodes each, with decision time | 5 |
| `mismatch` | Dependence on known physics | true MEC capacity −20 %, PA efficiency 0.25 instead of 0.35, cycles per bit +20 %, max-min fair instead of proportional MEC sharing; arms: SFR-SAC with wrong constants, SFR-SAC with correct constants, Safe TD3 (model-free), MB-greedy (full information) | 3 |
| `cpo` | CPO baseline (R2.C4, E7) | CPO with the split service-flow actor, γ = 0 and γ = 0.99, N = 50…70 | 5 |
| `horizon` | Immediate vs discounted critics (R1.1, E1) | Safe SAC with MLP critics, Safe TD3, Penalty SAC β=5; γ = 0 and 0.99 | 3 |
| `penalty_form` | Hinge vs untruncated penalty (R1.2, E2) | SFR-SAC hinge and linear, d ∈ {8, 10, 12} % (slot P95 failure reported) | 3 |
| `ap` | AP-dependent gains (R1.4, E3) | SFR-SAC with learned AP, best-channel AP, AP-independent gains; Safe TD3 and MB-greedy with both gain models | 3 |
| `latency` | Decision latency (R1.7, E6) | one CPU thread: SFR-SAC and Safe TD3 actors at N = 50, 70, 200, and the trained N = 70 actor; per-slot optimizers' decision times come from `per_slot` | – |

The structured critics only predict immediate outcomes, so E1 is run with generic critics, where a
discount factor is meaningful. Both arms of each E1 pair use the sampled AP-entropy term, because the
exact entropy controls require γ = 0.

## Running it

Requirements are the same as for the main study (CUDA GPU server). CPO, MB-greedy, the per-slot
optimizers and the latency benchmark run on CPU workers.

```bash
python -m pytest -q tests/test_revision.py          # about 10 s
bash run_revision.sh smoke                           # CPU rehearsal of every stage, about 10 min
bash run_revision.sh dry-run                         # plan and GPU-time estimate
bash run_revision.sh start --gpus 0,1 --workers 8 --cpu-workers 8
bash run_revision.sh status
bash run_revision.sh stop                            # graceful; `start` again resumes from checkpoints
bash run_revision.sh report                          # outputs/revision/report/REVISION_REPORT.md + CSV/LaTeX
```

Plan: 175 trainings (107 on GPU, about 93 SFR-SAC-equivalents; 68 on CPU: 50 CPO and 18 MB-greedy),
175 held-out tests, 20 per-slot optimizer runs, and one latency benchmark. With 8 concurrent GPU
runs at the main study's 2.5 h per SFR-SAC run this is about 29 h of GPU time; the CPU tasks run in
parallel. Stages run in the order of the table (most important first); `--stage <name>` runs one.
Identical configurations are trained once and shared (for example, the SFR-SAC reference of
`fair_critics`, `per_slot`, `penalty_form` d = 10 %, `ap` and `mismatch`).

Every arm in a stage is tested on the same held-out episodes, using the main study's stream
`2900000 + 100 N + layout − 1000` with 100 episodes (per-slot optimizers: the first 30). The output
directory records both the frozen main-study digest and a revision-extension digest. A task is
incomplete when either digest does not match, so results from different extension versions are not
mixed. The historical main-study digest remains separate and is never rewritten.

## Decision rules (fix before looking at the results)

* `fair_critics`: compare paired energy and failure differences at the declared failure-matching
  rule. A confidence interval containing zero is inconclusive; it is not evidence of equivalence.
  Equivalence requires a pre-specified margin and an equivalence test. If no equivalence test is
  run, report the paired estimate and interval without assigning a winner.
* `per_slot`: if descent on the trained critics from a neutral start matches the actor, the actor's
  value is amortization; report the decision-time ratio (one actor pass is about 1 ms at N = 70 on
  one CPU thread, the critic descent hundreds of ms). The oracle bounds what full information gives.
* `mismatch`: report the loss of SFR-SAC with wrong constants against SFR-SAC with correct ones and
  against Safe TD3. The full-information MB-greedy arm is a model-aware reference, not a learned
  baseline.

## Code changes outside `revision/`

All are optional; with default settings every main-grid method is bitwise identical to `main`
(weights and evaluation), which `python -m revision.regression_check <main checkout> .` verifies.
The frozen `original_ch5` results can therefore be tabulated next to the revision results.

* `agents/td3_agent.py`: optional structured reward/cost critics (same losses and gradient clipping
  as SFR-SAC, including the per-device delivery and energy labels) and an optional policy network,
  wrapped by `DeterministicPolicyAdapter` (tanh for x and p, per-ED softmax over APs).
* `v2/cpo.py`: optional policy network (the split service-flow actor). **Fix:** the cost advantage is
  no longer multiplied by (1 − γ_c). `J_c` is the per-step average cost, whose gradient is
  E[ratio · A_c] with the advantage itself; the factor shrank the constraint gradient 100× at
  γ_c = 0.99 (no effect at γ_c = 0). CPO is not part of `results/final`.
* `v2/runtime.py`: `model_assumptions` (physical constants given to the structured actor and
  critics instead of the simulator's); TD3 and CPO network options; `load_agent` no longer treats
  CPO's GAE `lam` as the SAC multiplier, and takes `check_source=False` for old checkpoints.
* `v2/env.py`: `mec_sharing="max_min_fair"` option (default proportional) for the mismatch study.
* `agents/cloud_actor.py`: `ap_rule="max_gain"` executes the best-channel AP (E3).
* `cloud24/worker.py`: held-out tests of CPO models.
* `scripts/evaluate_model.py`: `--allow-source-mismatch` to evaluate the shipped example model.
* `docs/REPRODUCIBILITY.md`: the state dimension is 4N + M + 8 + NM (714 at N = 70), not 3N + M + 8.

## Notes for the main branch

* The revision branch intentionally changes the frozen main-study source digest because it adds
  optional critics, CPO, model assumptions, and AP/environment controls. It must not be used to
  resume the historical `original_ch5` runs or to load their checkpoints. Use a separate checkout
  of the historical main commit for the original study; use this branch only for the revision
  stages. The shipped example model remains a historical artifact and is not used as a revision
  parent.
* `configs/base.json` requires CUDA; the CPU commands in the README work only through
  `scripts/train_one.py`, which overrides the device.
* Not addressed here: per-device reliability. At N = 70 SFR-SAC's device-failure P95 is 27.9 %,
  the highest of the constrained methods (TD3 20.2 %, MB-greedy 13.2 %). That needs a method change
  (for example a per-device or CVaR term) and is a decision for the authors.
