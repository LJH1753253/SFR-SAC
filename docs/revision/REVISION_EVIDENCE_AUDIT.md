# IoT-J Revision Evidence Audit

**Audit date:** 2026-11-09  
**Scope:** read-only consistency and mechanism audit for the IoT-J supplementary experiments.  
**Repository:** SFR-SAC  

This audit does not retrain policies, change algorithm code, or overwrite historical outputs. The diagnostic script and its generated files are in [`diagnostics/revision_evidence_audit_20261109.py`](../../diagnostics/revision_evidence_audit_20261109.py) and [`diagnostics/revision_evidence_audit_20261109/`](../../diagnostics/revision_evidence_audit_20261109/).

## Executive conclusion

1. **No P0 raw-result corruption was found.** The reported CSV files, summary statistics, and AP-load diagnostics are internally consistent.
2. **The old 8.59 mJ result and the new 9.68 mJ result are not an exact same-source replication.** The evaluation stream, training length, nominal environment settings, and paired layouts match, but the source digest, hardware, and one explicit configuration field differ. The new mean is almost entirely explained by seed 0; the other four paired seeds are essentially unchanged.
3. **The AP-independent 39.99% result is a real model-behaviour result, not a reporting bug.** With AP-independent gains, the association problem is symmetric. The learned policy concentrates traffic on two APs, creating severe load imbalance. This is a valid negative control, but it should be described as an identifiability/symmetry limitation rather than as an algorithm failure caused by a code error.
4. **The existing fixed-penalty sweep does not provide a same-failure-rate energy match to SFR-SAC.** It supports a cost-energy trade-off curve, but not a claim that fixed penalty achieves lower energy at the same service-failure level. A targeted beta sweep below 2 would be required for that claim.

## 1. Old versus Fair-Critics SFR-SAC

### 1.1 Paired results

The following values use the same five paired layouts/seeds, with the difference defined as `Fair - Original`. Cost and BFSR are percentage points; energy is mJ/device/slot.

| Seed | Layout | Original cost | Fair cost | Delta cost | Original energy | Fair energy | Delta energy | Original BFSR | Fair BFSR | Delta BFSR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1000 | 9.325 | 9.436 | +0.111 | 8.190 | 13.604 | +5.414 | 96.390 | 65.800 | -30.590 |
| 1 | 1001 | 9.514 | 9.560 | +0.046 | 8.789 | 8.781 | -0.008 | 93.840 | 91.335 | -2.505 |
| 2 | 1002 | 9.419 | 9.354 | -0.065 | 8.258 | 8.241 | -0.017 | 97.110 | 97.805 | +0.695 |
| 3 | 1003 | 9.128 | 9.217 | +0.089 | 8.720 | 8.707 | -0.013 | 99.535 | 99.555 | +0.020 |
| 4 | 1004 | 9.516 | 9.357 | -0.159 | 9.015 | 9.054 | +0.039 | 94.270 | 96.690 | +2.420 |
| **Mean +/- sample SD** |  | **9.38 +/- 0.16** | **9.38 +/- 0.13** |  | **8.59 +/- 0.36** | **9.68 +/- 2.21** |  | **96.23 +/- 2.31** | **90.24 +/- 14.00** |  |

The energy difference is not a broad shift across all seeds. Seed 0 contributes +5.414 mJ; seeds 1--4 differ by -0.017, -0.013, +0.039, and -0.008 mJ, respectively. Therefore, the correct interpretation is:

> The new Fair-Critics run reproduces the original operating point for four of five paired cases, while seed 0 exhibits a distinct training outcome with poorer AP balancing and lower offloading. The two aggregate means should not be treated as two independent estimates from an identical implementation.

### 1.2 What is and is not matched

| Item | Original archive | Fair-Critics run | Audit result |
|---|---|---|---|
| Training episodes | 1600 | 1600 | Matched |
| Final evaluation stream | `2907000` for seed 0, with the same layout-derived stream | `2907000` for seed 0 | Matched |
| Paired layouts | 1000--1004 | 1000--1004 | Matched |
| Nominal eta, d, background load | 0.30, 0.10, 0.05 | 0.30, 0.10, 0.05 | Matched |
| Physical energy and critic setting | Physical energy, gamma=0 | Physical energy, gamma=0 | Matched in recorded configs |
| Explicit MEC sharing field | Key absent; original environment default is proportional | `mec_sharing="proportional"` | Semantically matched, not byte-identical |
| Source digest | `e942ce...` | `36c128...` | Not identical |
| GPU | A800-SXM4-80GB | RTX 4070 SUPER | Not identical |
| Model checkpoint identity | Not available in the compact archive | Fresh Fair-Critics training | Not established as identical |

The old source digest is not present in the current checkout, so exact source-level equivalence cannot be reconstructed from the results archive. The new run is therefore suitable as a Fair-Critics result, but it is not a bitwise re-run of the old result.

### 1.3 Why seed 0 is different

The seed-0 paired diagnostics show a coherent mechanism-level change:

| Diagnostic | Original seed 0 | Fair seed 0 |
|---|---:|---:|
| Average offload ratio | 0.4749 | 0.2719 |
| Mean SNR (dB) | 16.394 | 5.599 |
| AP-load coefficient of variation | 0.467 | 1.593 |
| Cost (%) | 9.325 | 9.436 |
| Energy (mJ/device/slot) | 8.190 | 13.604 |
| BFSR (%) | 96.390 | 65.800 |

The changed seed-0 policy is not merely a summary or rounding anomaly: its offloading, SNR, and AP-load statistics also move. The available evidence does not identify whether the cause is the source revision, hardware-dependent optimization trajectory, or a fresh-training basin. It does establish that the discrepancy is concentrated in one paired training outcome.

## 2. AP-independent channel diagnostic

### 2.1 Relevant implementation

When `ap_dependent_gain=False`, [`v2/env.py`](../../v2/env.py#L177-L181) replaces every ED's six AP gains with copies of that ED's best-link scalar:

```python
g = self.large_scale_db + self.fading_db
if not self.params["ap_dependent_gain"]:
    g = np.repeat(g[np.arange(self.N), self.best_ap_static][:, None], self.R, axis=1)
```

The environment then decodes the AP branch using an argmax over the reshaped ED-by-AP action block at [`v2/env.py`](../../v2/env.py#L251-L255). The AP load statistic is computed from the selected AP counts at [`v2/metrics.py`](../../v2/metrics.py#L27-L31). The AP-independent variant is enabled explicitly in [`revision/variants.py`](../../revision/variants.py#L91-L98).

### 2.2 Observed mechanism

The formal AP-independent summary is approximately 39.99% cost with very low BFSR. The read-only seed-0 diagnostic used five episodes and 1,000 observed slots and found:

| Quantity | AP-independent SFR | AP-dependent SFR | Max-gain AP rule |
|---|---:|---:|---:|
| Mean cost (%) | 40.173 | 9.327 | 9.649 |
| Mean AP-load CV | 2.236 | 0.473 | 0.236 |
| Selected AP counts | 35,000 / 35,000 / 0 / 0 / 0 / 0 | 9,061 / 9,557 / 8,620 / 7,152 / 23,178 / 12,432 | 12,516 / 15,355 / 11,703 / 12,882 / 10,306 / 7,238 |
| Tie-heavy slot fraction | 0.005 | 0 | 0 |

The result is therefore **not an all-AP argmax tie bug**. The learned policy selects two APs consistently, while the remaining four receive no traffic. Equal channel gains remove the link-based information that can distinguish APs; the AP head has no explicit load-balancing objective in this variant. The policy can consequently break the symmetric action space arbitrarily and settle on a concentrated partition.

This negative control is useful in the paper only if presented as a limitation or sanity check:

> AP association requires AP-discriminating information. When all AP gains are forced to be identical for each ED, the association problem becomes underdetermined and the learned policy may concentrate traffic on a subset of APs.

It should not be presented as evidence of a software defect, nor should the 5-episode diagnostic replace the formal AP-independent evaluation.

## 3. Fixed-penalty matching

### 3.1 Current implementation

For `pen_struct_b<beta>`, [`revision/variants.py`](../../revision/variants.py#L102-L108) sets:

```python
penalty_mode="linear"
lambda_mode="fixed"
fixed_lambda=beta / safe["cost_coef"]
initial_lambda=beta / safe["cost_coef"]
```

The fixed branch is enforced in [`agents/safe_sac_agent.py`](../../agents/safe_sac_agent.py#L393-L408) and [`agents/safe_sac_agent.py`](../../agents/safe_sac_agent.py#L476-L489). It is a genuine fixed-penalty policy, not an adaptive dual run with a renamed parameter.

### 3.2 Existing Fair-Critics results

| Fixed beta | Cost (%) | Energy (mJ/device/slot) | BFSR (%) |
|---:|---:|---:|---:|
| 2.0 | 8.51 +/- 0.50 | 8.75 +/- 0.22 | 75.13 +/- 7.00 |
| 2.5 | 5.06 +/- 0.40 | 10.26 +/- 0.51 | 99.52 +/- 0.31 |
| 3.0 | 4.22 +/- 0.86 | 10.58 +/- 0.82 | 99.87 +/- 0.12 |
| 3.5 | 3.73 +/- 0.56 | 10.96 +/- 0.63 | 99.96 +/- 0.04 |
| 4.5 | 4.22 +/- 1.59 | 11.68 +/- 0.93 | 99.48 +/- 0.86 |
| **SFR-Critic reference** | **9.38 +/- 0.13** | **9.68 +/- 2.21** | **90.24 +/- 14.00** |

Every tested fixed-penalty mean has lower cost than the SFR-Critic reference, but no value is close enough to establish a same-cost energy comparison. The existing table is valid as a **trade-off curve**: higher penalty pressure reduces failure and increases energy over the tested range. It does not support “fixed penalty uses less energy at the same failure rate.”

If that equal-cost comparison is important, use the same structured protocol and run a targeted beta grid below 2, for example `{1.6, 1.7, 1.8, 1.9, 2.0}`. Use at least three paired seeds for a probe and five for a final table. Do not interpolate a directly trained policy and label it as an experimental point.

## 4. Evidence safe for the IoT-J paper

### Safe to use as primary evidence

1. **Fair-Critics main comparison** with its own five-pair mean and standard deviation, clearly labelled as the revised implementation.
2. **Fair-Critics versus TD3 or other baselines** when both rows use the same Fair protocol, source, layout stream, and evaluation procedure.
3. **AP-dependent association ablation**, including the learned AP branch and the max-gain rule, as evidence that AP-dependent link information improves association balance.
4. **Latency results**, provided the hardware and timing protocol are stated.
5. **Penalty-form and horizon/mismatch results** as sensitivity or calibration evidence, not as proof of a universal ranking.

### Do not combine or overstate

1. Do not pool the old 8.59 mJ mean with the new 9.68 mJ mean as if they were repetitions of one identical implementation.
2. Do not use AP-independent 39.99% to claim that SFR-SAC generally fails; use it only as the symmetry/identifiability negative control.
3. Do not claim fixed penalty is less energy-consuming at the same failure cost without the targeted beta sweep.
4. Do not claim a per-device reliability guarantee from mean cost or BFSR alone.
5. Do not describe the immediate critic setting as a long-horizon CMDP guarantee. The recorded study uses gamma=0; it is an immediate-risk predictor combined with empirical dual regulation.

## 5. Priority actions

### P0: none for data integrity

No P0 corruption or file-mapping error was found. Formal outputs were not modified by the audit.

### P1: required before final paper claims

1. State that the Fair-Critics run is the revised protocol and report its own statistics; do not call it an exact reproduction of the old 8.59 mJ result.
2. Add a short limitation explaining why AP-independent gains make association underdetermined and can cause concentrated AP usage.
3. Label the fixed-penalty table as an energy-failure trade-off unless the targeted beta matching experiment is completed.
4. Refresh the provenance/status description. The repository's current `docs/` tree does not contain `docs/revision/SUPPLEMENTARY_EXPERIMENT_LOG.md`; the corresponding log exists only in the separate overnight workspace. The current result package remains usable, but the supplementary log should be copied in deliberately if it is required for submission.

### P2: optional strengthening

1. Run the targeted beta grid `{1.6, 1.7, 1.8, 1.9, 2.0}` with five paired seeds if an equal-failure energy claim is central.
2. Repeat the AP-independent negative control with the formal evaluation length if the reviewer asks for a full-scale robustness check. The current 5-episode diagnostic is mechanism evidence, not a replacement for the formal table.
3. Report the Fair seed-0 anomaly as a paired-training variability case and include per-seed values in supplementary material.

## 6. Reproducibility files

- Full revision results: [`results/revision_20261108/`](../../results/revision_20261108/)
- Selected evidence narrative: [`STRONG_EVIDENCE_SUMMARY.md`](../../results/revision_20261108/STRONG_EVIDENCE_SUMMARY.md)
- Fair-Critics local report: [`fair_critics_local_report.md`](../../results/revision_20261108/data/fair_critics_local_report.md)
- Paired old/new diagnostic: [`original_vs_fair_sfr.csv`](../../diagnostics/revision_evidence_audit_20261109/original_vs_fair_sfr.csv)
- Diagnostic JSON: [`diagnostics.json`](../../diagnostics/revision_evidence_audit_20261109/diagnostics.json)

