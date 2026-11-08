# Revision study report

Source digest `36c1289d0f1a5bba`; revision digest `7e9709f3b0c2743d`. Means ± sample SD over paired deployments; paired deltas use nominal 95 % t intervals. A CI containing zero is reported as inconclusive evidence, not equivalence.

## fair_critics

| Variant | N | Pairs | Failure (%) | Energy (mJ) | BFSR (%) | Device P95 (%) | Power (W) | Final lambda | d Energy vs sfr | d Failure vs sfr |
|---|---|---|---|---|---|---|---|---|---|---|
| pen_struct_b2 | 70 | 3 | 8.51±0.50 | 8.75±0.22 | 75.13±7.00 | 28.06±5.20 | 0.14±0.00 | 0.80±0.00 | -1.46 [-9.09, +6.16] | -0.94 [-2.01, +0.13] |
| pen_struct_b2.5 | 70 | 3 | 5.06±0.40 | 10.26±0.51 | 99.52±0.31 | 11.93±1.26 | 0.14±0.00 | 1.00±0.00 | +0.05 [-7.43, +7.52] | -4.39 [-5.63, -3.16] |
| pen_struct_b3 | 70 | 3 | 4.22±0.86 | 10.58±0.82 | 99.87±0.12 | 11.33±4.34 | 0.14±0.00 | 1.20±0.00 | +0.37 [-8.38, +9.12] | -5.23 [-7.54, -2.91] |
| pen_struct_b3.5 | 70 | 3 | 3.73±0.56 | 10.96±0.63 | 99.96±0.04 | 10.43±2.91 | 0.14±0.00 | 1.40±0.00 | +0.75 [-7.13, +8.63] | -5.72 [-7.35, -4.10] |
| pen_struct_b4.5 | 70 | 3 | 4.22±1.59 | 11.68±0.93 | 99.48±0.86 | 12.16±6.21 | 0.14±0.00 | 1.80±0.00 | +1.48 [-4.06, +7.01] | -5.23 [-9.29, -1.18] |
| sfr | 70 | 5 | 9.38±0.13 | 9.68±2.21 | 90.24±14.00 | 27.66±5.71 | 0.14±0.00 | 3.02±0.13 | -- | -- |
| td3 | 70 | 5 | 9.18±0.41 | 11.02±0.70 | 75.80±7.69 | 18.59±0.46 | 0.12±0.01 | 6.67±0.77 | +1.34 [-1.77, +4.45] | -0.21 [-0.72, +0.30] |
| td3_struct | 70 | 5 | 9.54±0.12 | 8.83±0.32 | 70.72±4.60 | 23.00±4.04 | 0.19±0.00 | 2.80±0.13 | -0.85 [-3.90, +2.21] | +0.15 [-0.09, +0.39] |
| td3_struct_split | 70 | 5 | 9.69±0.20 | 8.97±1.46 | 77.34±14.75 | 25.50±2.64 | 0.17±0.04 | 2.81±0.12 | -0.71 [-4.36, +2.95] | +0.30 [-0.02, +0.63] |

Matched-failure comparison with the structured fixed-penalty sweep: not available yet.

## mismatch

| Variant | N | Pairs | Failure (%) | Energy (mJ) | BFSR (%) | Device P95 (%) | d Energy vs nominal_sfr | d Failure vs nominal_sfr |
|---|---|---|---|---|---|---|---|---|
| nominal_sfr | 70 | 3 | 9.45±0.10 | 10.21±2.95 | 84.98±16.92 | 27.38±6.94 | -- | -- |
| nominal_td3 | 70 | 3 | 9.36±0.27 | 10.67±0.26 | 72.55±5.81 | 18.43±0.50 | +0.46 [-7.16, +8.08] | -0.09 [-0.79, +0.60] |

## horizon

| Variant | N | Pairs | Failure (%) | Energy (mJ) | BFSR (%) | Offload x |
|---|---|---|---|---|---|---|
| td3_g0 | 70 | 3 | 9.36±0.27 | 10.67±0.26 | 72.55±5.81 | 0.47±0.01 |

## penalty_form

| Variant | N | Pairs | Failure (%) | Slot P95 (%) | Energy (mJ) | BFSR (%) | Device P95 (%) | Final lambda | d Energy vs sfr_hinge_d0.1 | d Failure vs sfr_hinge_d0.1 |
|---|---|---|---|---|---|---|---|---|---|---|
| sfr_hinge_d0.1 | 70 | 3 | 9.45±0.10 | 10.82±1.44 | 10.21±2.95 | 84.98±16.92 | 27.38±6.94 | 2.94±0.09 | -- | -- |

## ap

| Variant | N | Pairs | Failure (%) | Energy (mJ) | Best AP (%) | AP-load CV | d Energy vs sfr | d Failure vs sfr |
|---|---|---|---|---|---|---|---|---|
| sfr | 70 | 3 | 9.45±0.10 | 10.21±2.95 | 70.95±32.70 | 0.65±0.82 | -- | -- |
| td3 | 70 | 3 | 9.36±0.27 | 10.67±0.26 | 80.12±0.59 | 0.17±0.03 | +0.46 [-7.16, +8.08] | -0.09 [-0.79, +0.60] |

Completed cases: 53 of 217.

