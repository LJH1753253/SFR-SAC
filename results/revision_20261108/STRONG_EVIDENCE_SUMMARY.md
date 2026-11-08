# SFR-SAC Revision: Strong-Evidence Summary

## Scope

This summary highlights the experiments that directly answer the main reviewer questions. The complete raw summaries and LaTeX tables remain in `data/` and `tables/`.

The principal policy is SFR-SAC. Energy is reported in mJ/device/slot; service-failure cost and BFSR are percentages.

## 1. Fair Critic Comparison

The local Fair Critics experiment used five paired deployments at `N=70`.

| Policy | Cost (%) | Energy (mJ/device/slot) | BFSR (%) |
|---|---:|---:|---:|
| SFR-SAC | 9.38 +/- 0.13 | 9.68 +/- 2.21 | 90.24 +/- 14.00 |
| Safe TD3 | 9.18 +/- 0.41 | 11.02 +/- 0.70 | 75.80 +/- 7.69 |
| TD3 + Structured Critics | 9.54 +/- 0.12 | 8.83 +/- 0.32 | 70.72 +/- 4.60 |
| TD3 + Structured Critics + Split Actor | 9.69 +/- 0.20 | 8.97 +/- 1.46 | 77.34 +/- 14.75 |

The strongest baseline comparison is ordinary Safe TD3: SFR-SAC uses approximately 12.2% less energy and improves BFSR by approximately 14.4 percentage points at a comparable mean service-failure cost.

The architecture-matched comparison is TD3 + Structured Critics + Split Actor. SFR-SAC improves BFSR by approximately 12.9 percentage points and reduces mean cost by approximately 0.30 percentage points, with an energy increase of approximately 0.71 mJ/device/slot. This isolates the constrained SAC mechanism more directly than an unmatched network comparison.

## 2. AP-Dependent Channel Evidence

The AP ablation provides the clearest modeling result:

| SFR-SAC variant | Cost (%) | Energy (mJ/device/slot) |
|---|---:|---:|
| AP-dependent channel gain | 9.42 +/- 0.09 | 8.41 +/- 0.33 |
| AP-independent channel gain | 39.99 +/- 0.26 | 8.73 +/- 0.34 |

Removing AP-dependent channel gains increases the service-failure cost from 9.42% to 39.99%. This shows that the AP-association branch is not merely a load-balancing proxy; channel-aware association is essential to the modeled problem.

The best-channel AP control has a similar mean cost (9.46%) but slightly higher energy (8.61 mJ/device/slot), indicating that the learned association can retain the reliability level while allowing a different energy/load operating point.

## 3. Online Decision Latency

On one Intel Xeon Platinum 8358P CPU thread, the SFR-SAC policy/action-decode path remains below 2 ms through `N=200`:

| N | Median (ms) | P95 (ms) | P99 (ms) |
|---:|---:|---:|---:|
| 50 | 1.211 | 1.230 | 1.239 |
| 70, initialized | 1.223 | 1.278 | 1.285 |
| 70, trained | 1.231 | 1.285 | 1.293 |
| 200 | 1.538 | 1.558 | 1.566 |

These values support an amortized-optimization interpretation: the actor transfers most optimization effort to offline training and produces a low-latency online decision. The measurement covers policy inference and action decoding, not state acquisition, signaling, or actuator overhead.

## 4. AP and Physical-Model Robustness

The model-mismatch results identify which physical assumptions matter most:

- With a correct MEC-capacity assumption, SFR-SAC obtains 9.33% cost and 97.70% BFSR; with a 20% capacity mismatch, cost rises to 13.76% and BFSR falls to 2.80%.
- With a correct cycles-per-bit assumption, SFR-SAC obtains 9.29% cost and 98.38% BFSR; with a 20% cycles-per-bit mismatch, cost rises to 11.99% and BFSR falls to 11.95%.
- Replacing proportional MEC sharing with max-min sharing has a much smaller effect: 9.41% cost and 95.35% BFSR for the mismatched structured policy.

The direct conclusion is that the structured physical prior is useful when calibrated, while MEC capacity and CPU intensity must be specified accurately.

## 5. Immediate and Discounted Critics

The clearest horizon result is obtained for the fixed-penalty SAC control:

| Variant | Cost (%) | Energy (mJ/device/slot) | BFSR (%) |
|---|---:|---:|---:|
| Penalty SAC, gamma=0 | 5.31 +/- 0.72 | 11.69 +/- 0.84 | 99.41 +/- 0.46 |
| Penalty SAC, gamma=0.99 | 11.43 +/- 0.70 | 12.74 +/- 1.36 | 29.45 +/- 8.83 |

Under the one-shot slot-completion setting, the discounted target does not provide a general improvement and can substantially degrade the constrained operating point. This supports retaining the immediate slot-cost critic for the main one-slot formulation.

## 6. Penalty-Form Stability

At the strict budget setting `d=0.08`, the hinge form is substantially more stable than the linear form:

| Variant | Cost (%) | Energy (mJ/device/slot) | BFSR (%) |
|---|---:|---:|---:|
| Hinge, d=0.08 | 7.19 +/- 0.09 | 9.43 +/- 0.29 | 97.22 +/- 0.95 |
| Linear, d=0.08 | 7.78 +/- 3.86 | 9.08 +/- 1.37 | 52.97 +/- 45.38 |

The hinge penalty provides a much more stable constrained operating point. The complete linear and hinge sweeps are retained in `tables/penalty_form.csv` and `data/REVISION_REPORT.md`.

## Main Conclusion

The revision evidence supports the following paper-level conclusion:

> SFR-SAC provides a strong constrained operating point rather than simply minimizing energy in isolation. Under both conventional and architecture-matched TD3 comparisons, it achieves substantially higher BFSR and comparable or lower service-failure cost, while retaining millisecond-level online inference. The AP-dependent channel ablation confirms that the association branch captures meaningful channel-aware decisions, and the mismatch study identifies the physical assumptions that must be calibrated for reliable structured control.

The repository retains the full experiment tables, raw stage summaries, protocol metadata, and provenance files for reproducibility.
