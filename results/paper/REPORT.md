# SFR-SAC Paper1 Experimental Report

## 1. Experimental Settings

**Scientific question.** Can SFR-SAC regulate aggregate service-failure cost while reducing device-side energy in a heterogeneous IIoT-MEC system?

**Setup.** The physical energy model combines CPU-cycle local energy with PA/circuit uplink energy. The main grid uses `N={50,55,60,65,70}`, five paired deployments, 1,600 training episodes, and a 200-slot episode horizon. Energy is reported in mJ/device/slot. Cost and BFSR are percentages. The revision controls use `eta=0.30`, `d=0.10`, `E_ref=0.02 J`, and `gamma=0`.

**Statistics.** A deployment seed/layout pair is the replication unit. Means and sample standard deviations are computed across completed pairs. Nominal paired 95% t intervals are retained where aligned per-seed rows exist. Slots are not treated as independent training replicates.

## 2. Main Baseline Comparison

**Question.** How does SFR-SAC compare with fixed-penalty SAC, Safe TD3, the Lyapunov-inspired rule, and Vanilla SAC across device scales?

**Results.** SFR-SAC stays near the 10% aggregate budget over all tested scales. At `N=70`, it obtains `9.38 +/- 0.16%` cost, `8.59 +/- 0.36 mJ/device/slot`, and `96.23 +/- 2.31%` BFSR. Safe TD3 obtains comparable cost, `9.28 +/- 0.37%`, but higher energy, `10.98 +/- 0.51 mJ`, and lower BFSR, `72.78 +/- 9.92%`. Fixed-penalty SAC reduces cost and increases energy. Vanilla SAC has low energy but severe service failure and zero BFSR in the main grid.

**Interpretation.** SFR-SAC provides a constrained operating point rather than an energy minimum without reliability regulation. Energy rankings are not interpreted independently of cost and BFSR.

## 3. Fair Architecture and Critic Comparison

**Question.** Which design component contributes to the operating point: structured critics, the split actor, or constrained SAC regulation?

**Setup.** The latest container run contains five-pair SFR-SAC, Safe TD3, TD3 with structured critics, and TD3 with structured critics plus a split actor. Fixed-penalty rows retain their recorded three-pair coverage.

**Results.** SFR-SAC obtains `9.38 +/- 0.16%` cost, `8.59 +/- 0.36 mJ`, and `96.23 +/- 2.31%` BFSR. Safe TD3 obtains `9.28 +/- 0.37%`, `10.98 +/- 0.51 mJ`, and `72.78 +/- 9.92%`. TD3 with structured critics obtains `9.35 +/- 0.08%`, `8.88 +/- 0.35 mJ`, and `76.71 +/- 3.67%`. TD3 with structured critics and split actor obtains `10.14 +/- 0.56%` cost and `56.67 +/- 15.41%` BFSR, exceeding the nominal budget on average.

**Interpretation.** Structured critics account for a substantial part of the energy operating point. The remaining comparison is consistent with an additional constrained stochastic-policy and dual-regulation effect, but the full improvement is not attributed to SAC alone. The split-actor TD3 result is not used for unconditional energy ranking because its average cost exceeds the budget.

## 4. Per-Slot Optimization

**Question.** Does the trained actor provide amortized decision making compared with optimizing the critics or the full simulator model at every slot?

**Results.** The Actor obtains `9.38 +/- 0.16%` cost, `8.59 +/- 0.38 mJ`, `96.37 +/- 2.12%` BFSR, and `1.750 +/- 0.013 ms` median decision time. Critic refine obtains `9.79 +/- 0.04%`, `8.36 +/- 0.39 mJ`, and `84.65 +/- 2.26%`, with `246.295 +/- 0.993 ms` median time. Critic scratch obtains `9.86 +/- 0.04%`, `8.53 +/- 0.35 mJ`, and `74.26 +/- 4.91%`, with `1172.968 +/- 1.245 ms`. The full-information numerical optimizer obtains `7.71 +/- 0.33 mJ`, but cost is `10.00 +/- 0.00%`, BFSR is `53.22 +/- 0.87%`, and median time is `130.488 +/- 1.318 ms`.

**Interpretation.** The actor transfers optimization effort to offline training and produces millisecond-level online decisions. The numerical optimizer is a full-information numerical baseline, not a mathematically proven global optimum. Its lower energy is not an equal-reliability advantage.

**Limitation.** Timing covers the measured policy or optimization path, not state sensing, signaling, or actuation.

## 5. Constraint Regulation and Penalty Analysis

**Question.** How do the budget, service-window fraction, fixed penalty, and background load affect the operating point?

**Results.** Tighter budgets reduce cost and generally increase energy: `d=0.08` gives `7.08 +/- 0.19%` cost and `9.61 +/- 0.39 mJ`, while `d=0.12` gives `11.41 +/- 0.26%` and `7.92 +/- 0.30 mJ`. The eta sweep is non-monotonic: `eta=0.30` gives `9.38 +/- 0.16%` cost and `8.59 +/- 0.36 mJ`, while very small or very large windows increase cost and energy. Background-load results stay close to the nominal point over the tested values.

The fixed-penalty sensitivity is a trade-off curve. Low beta produces high failure, while larger beta generally lowers cost and increases energy. All recorded beta rows, including high-variance cases, remain in the canonical tables.

**Qualification.** No fixed-penalty method is claimed to be energy-superior at exactly the SFR-SAC failure level. The earlier matching analysis is not used as a primary statistical claim when paired coverage is insufficient.

## 6. CPO Comparison

**Question.** How does a standard constrained-policy baseline behave under immediate and discounted settings across device scales?

**Results.** The CPO table reports all tested `gamma=0` and `gamma=0.99` rows for `N=50,...,70`. The rows exhibit different cost-energy-BFSR operating points; no single row is treated as a universal winner.

**Qualification.** This is a baseline comparison under the recorded protocol, not a claim that CPO and SFR-SAC use identical critics or implementation details.

## 7. Immediate vs. Discounted Critics

**Question.** Does a discount factor improve the tested generic-critic baselines?

**Results.** The horizon table includes SAC-MLP, Safe TD3, and Penalty SAC at `gamma=0` and `gamma=0.99`. Penalty SAC changes from `5.31 +/- 0.72%` cost, `11.69 +/- 0.84 mJ`, and `99.41 +/- 0.46%` BFSR at `gamma=0` to `11.43 +/- 0.70%`, `12.74 +/- 1.36 mJ`, and `29.45 +/- 8.83%` at `gamma=0.99`.

**Interpretation.** In the one-shot slot-completion setting, the discounted generic-critic variants do not provide a general empirical improvement.

**Limitation.** This ablation targets generic critics and is not a complete one-variable discount-factor experiment for every structured SFR-SAC component.

## 8. AP Association Ablation

**Question.** Under what conditions is the AP-association branch identifiable and useful?

**Results.** The AP-dependent learned SFR-SAC obtains `9.42 +/- 0.09%` cost in the recorded three-pair control. The AP-independent SFR-SAC obtains `39.99 +/- 0.26%` cost and AP-load CV `2.24`. The complete table also includes best-channel AP, AP-independent Safe TD3, and AP-independent MB-greedy controls.

**Interpretation.** AP association requires distinguishable candidate-AP information. When gains are forced to be symmetric, association can concentrate on a subset of APs and create resource contention. This is an applicability condition and model limitation, not a universal failure claim.

## 9. Physical Model Mismatch

**Question.** How sensitive is the structured policy to incorrect MEC capacity, PA efficiency, CPU cycles per bit, and MEC sharing assumptions?

**Results.** The mismatch table compares wrong and correct SFR assumptions, Safe TD3, and MB-greedy within each recorded perturbation. Wrong MEC-capacity and cycles-per-bit assumptions can exceed the 10% cost budget and sharply reduce BFSR. Correct assumptions improve feasibility, but energy remains dependent on the perturbed environment.

**Interpretation.** The structured physical prior is useful only when its assumed constants are calibrated to the simulator or deployment setting.

## 10. Computational Latency

**Question.** How does online policy inference scale with the number of devices?

**Results.** The one-thread benchmark includes `N=50,70,200` for SFR-SAC and Safe TD3, plus the trained SFR-SAC actor. SFR-SAC initialized-policy median latency is `1.211`, `1.223`, and `1.538 ms` for `N=50,70,200`; the trained `N=70` actor is `1.231 ms`. Safe TD3 is faster in this implementation, with median values below `0.4 ms` through `N=200`.

**Qualification.** These values compare implementation paths on the stated hardware. They are not full end-to-end system latency measurements.

## 11. Additional Ablation Studies

The canonical package retains eta sensitivity, budget sensitivity, background-load sensitivity, penalty beta sensitivity, lambda-update comparisons, failure composition, and Fair-rule comparisons under `tables/` and `per_seed/`. The plots expose the full tested ranges rather than selecting only favorable points. Lambda and dual-update results are implementation sensitivities, not universal optimality claims.

## 12. Statistical Analysis

Available paired results report mean and sample SD across deployment pairs. Paired differences and nominal 95% t intervals are retained where aligned per-seed rows exist. Intervals are exploratory and have no multiple-comparison correction. An interval containing zero is inconclusive evidence, not proof of equivalence.

## 13. Limitations

1. Aggregate cost and BFSR do not guarantee per-device reliability. Device-level P95 diagnostics can remain high.
2. The immediate cost-critic setting uses `gamma=0`; these experiments do not establish a long-horizon CMDP safety guarantee.
3. The AP-independent ablation is degenerate by construction because candidate AP gains are symmetric.
4. Numerical and MB-greedy baselines use simulator-model information unavailable to the learned actor.
5. Some revision controls use three paired deployments, while the container Fair-Critics and Per-slot rerun use five pairs. Counts are preserved and not silently pooled.
6. Timing excludes state sensing, communication, and actuation overhead.
