# Paper1 Revision Experiment Protocol

This document fixes the analysis before results are inspected. All stage
results use the revision branch and its revision source digest.

## Common Conditions

- Physical CPU-cycle local energy and PA/circuit TDMA-airtime energy.
- AP-dependent geometry, heterogeneous fixed CPUs, burst workloads, and
  one-shot slot completion.
- Default parameters: `eta=0.30`, `d=0.10`, `E_ref=0.02 J`, `gamma=0`.
- Training: 1,600 episodes, 200 slots per episode; final tests use 100
  episodes unless a stage declares otherwise.
- Main scale: `N={50,55,60,65,70}`.
- Five paired deployments: `(layout_seed, training_seed)=(1000+i,i)` for
  `i=0,...,4`; three-pair stages use the first three pairs.
- State: `4N+M+8+NM`; action: `2N+NM`.
- Primary metrics: mean service-failure cost (%) and mean device energy
  (mJ/device/slot). BFSR and device-level failure statistics are secondary.

## Stage Matrix

| Stage | Purpose | Main variables and arms | Pairs |
|---|---|---|---:|
| `fair_critics` | Separate critic, actor, and update-rule effects | SFR-SAC; Safe TD3; TD3 with SFR critics; TD3 with SFR critics and split actor; structured fixed penalties `beta={2,2.5,3,3.5,4.5}` | 5 |
| `per_slot` | Compare amortized actor decisions with direct optimization | trained actor; critic descent from actor; critic descent from neutral start; full-information per-slot optimizer | 5 |
| `mismatch` | Test sensitivity to assumed physics | MEC capacity -20%; PA efficiency 0.25; cycles/bit +20%; max-min MEC sharing; SFR wrong/right constants, TD3, MB-greedy | 3 |
| `cpo` | Add a mature constrained-RL baseline | CPO with split actor, `gamma=0` and `gamma=0.99`, `N=50...70` | 5 |
| `horizon` | E1 immediate versus discounted generic critics | SAC MLP critics, TD3, Penalty SAC; `gamma=0` and `0.99` | 3 |
| `penalty_form` | E2 hinge versus untruncated linear penalty | SFR-SAC hinge/linear at `d={0.08,0.10,0.12}` | 3 |
| `ap` | E3 AP/channel modeling | learned AP, best-channel AP, AP-independent gains; TD3 and MB-greedy controls | 3 |
| `latency` | E6 decision computation cost | SFR-SAC and TD3 at `N=50,70,200`, plus trained N=70 SFR-SAC | -- |

## Statistical Analysis

Each deployment pair is the unit of replication. Slots within a deployment are
not treated as independent training replicates. Report mean and sample SD over
completed pairs, with completed/expected counts shown for incomplete runs.

For comparisons with a paired reference, compute the per-pair difference using
the same layout and training seed, then report the mean difference and nominal
two-sided 95% t interval. A confidence interval containing zero means that the
current sample does not establish a detectable difference. It does not prove
equivalence. Equivalence requires a predeclared margin and a two-one-sided
equivalence test; no such claim is made unless that test is added before
looking at results.

Reliability is a prerequisite for energy comparisons: first report whether the
aggregate cost is within the declared budget and then compare energy. BFSR is
the fraction of test slots with cost at or below `d`; it is not a per-device
guarantee. Device P95 and the number of devices above budget remain separate
diagnostics.

The structured fixed-penalty sweep may be interpolated to the SFR-SAC failure
level only as a declared episode-level time-sharing analysis. It is not a
directly trained fixed-penalty policy at that exact failure level.

## Interpretation Boundaries

- Critic, actor, and RL-update contributions are reported separately.
- The per-slot optimizer is a finite numerical reference, not a global optimum
  or theoretical lower bound.
- E1 uses generic critics because the structured critics predict immediate
  physical outcomes; it is not a one-factor ablation of full SFR-SAC.
- E6 reports actor/action-decode computation only. State acquisition,
  signaling, simulator stepping, and other system latency are excluded.
- Failed, incomplete, or source-mismatched runs are excluded from means and
  listed in the report.
