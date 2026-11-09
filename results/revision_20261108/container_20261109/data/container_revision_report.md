# Revision study report

Source digest `36c1289d0f1a5bba`; revision digest `7e9709f3b0c2743d`. Means ± sample SD over paired deployments; paired deltas use nominal 95 % t intervals. A CI containing zero is reported as inconclusive evidence, not equivalence.

## fair_critics

| Variant | N | Pairs | Failure (%) | Energy (mJ) | BFSR (%) | Device P95 (%) | Power (W) | Final lambda | d Energy vs sfr | d Failure vs sfr |
|---|---|---|---|---|---|---|---|---|---|---|
| pen_struct_b2 | 70 | 3 | 10.35±3.25 | 8.99±0.42 | 52.74±41.28 | 30.53±6.04 | 0.14±0.01 | 0.80±0.00 | +0.58 [-0.71, +1.86] | +0.93 [-7.35, +9.22] |
| pen_struct_b2.5 | 70 | 3 | 5.14±0.45 | 10.24±0.52 | 99.37±0.42 | 12.24±1.30 | 0.14±0.00 | 1.00±0.00 | +1.83 [+1.09, +2.57] | -4.28 [-5.56, -3.00] |
| pen_struct_b3 | 70 | 3 | 5.57±2.75 | 11.65±1.52 | 92.06±13.65 | 13.67±8.02 | 0.14±0.00 | 1.20±0.00 | +3.24 [-0.82, +7.30] | -3.85 [-10.92, +3.21] |
| pen_struct_b3.5 | 70 | 3 | 3.59±0.54 | 10.89±0.62 | 99.96±0.04 | 9.16±1.39 | 0.14±0.00 | 1.40±0.00 | +2.48 [+1.69, +3.27] | -5.83 [-7.23, -4.42] |
| pen_struct_b4.5 | 70 | 3 | 3.44±0.47 | 11.16±0.59 | 99.98±0.03 | 10.14±2.90 | 0.14±0.00 | 1.80±0.00 | +2.75 [+2.04, +3.46] | -5.98 [-7.36, -4.59] |
| sfr | 70 | 5 | 9.38±0.16 | 8.59±0.36 | 96.23±2.31 | 27.85±5.03 | 0.14±0.00 | 3.03±0.13 | -- | -- |
| td3 | 70 | 5 | 9.28±0.37 | 10.98±0.51 | 72.78±9.92 | 20.19±1.84 | 0.12±0.01 | 6.79±1.01 | +2.38 [+2.05, +2.72] | -0.10 [-0.59, +0.39] |
| td3_struct | 70 | 5 | 9.35±0.08 | 8.88±0.35 | 76.71±3.67 | 22.90±4.26 | 0.19±0.00 | 2.80±0.13 | +0.29 [+0.08, +0.49] | -0.03 [-0.25, +0.20] |
| td3_struct_split | 70 | 5 | 10.14±0.56 | 10.57±2.96 | 56.67±15.41 | 27.73±6.22 | 0.19±0.03 | 2.69±0.17 | +1.98 [-1.37, +5.33] | +0.76 [+0.10, +1.41] |

SFR-SAC energy minus structured fixed-penalty energy at SFR-SAC's failure level (time-sharing between neighbouring beta values; betas [2.0, 2.5, 3.0, 3.5, 4.5]): -4.65 ± 0.00 mJ over 1 pairs (negative = SFR-SAC better).

## per_slot

| Variant | N | Pairs | Failure (%) | Energy (mJ) | BFSR (%) | Decision (ms) | Decision P95 (ms) | d Energy vs actor | d Failure vs actor |
|---|---|---|---|---|---|---|---|---|---|
| actor | 70 | 5 | 9.38±0.16 | 8.59±0.38 | 96.37±2.12 | 1.75±0.01 | 1.92±0.01 | -- | -- |
| critic_refine | 70 | 5 | 9.79±0.04 | 8.36±0.39 | 84.65±2.26 | 246.30±0.99 | 247.71±1.00 | -0.23 [-0.29, -0.18] | +0.41 [+0.19, +0.63] |
| critic_scratch | 70 | 5 | 9.86±0.04 | 8.53±0.35 | 74.26±4.91 | 1172.97±1.25 | 1197.67±3.04 | -0.06 [-0.21, +0.09] | +0.48 [+0.27, +0.69] |
| oracle | 70 | 5 | 10.00±0.00 | 7.71±0.33 | 53.22±0.87 | 130.49±1.32 | 131.43±1.34 | -0.88 [-1.03, -0.74] | +0.62 [+0.41, +0.82] |

## mismatch

| Variant | N | Pairs | Failure (%) | Energy (mJ) | BFSR (%) | Device P95 (%) | d Energy vs nominal_sfr | d Failure vs nominal_sfr |
|---|---|---|---|---|---|---|---|---|
| nominal_sfr | 70 | 3 | 9.42±0.09 | 8.41±0.33 | 95.78±1.72 | 27.45±5.88 | -- | -- |
| nominal_td3 | 70 | 3 | 9.30±0.52 | 10.77±0.54 | 71.43±13.31 | 19.21±0.88 | +2.36 [+1.47, +3.25] | -0.12 [-1.48, +1.23] |

## horizon

| Variant | N | Pairs | Failure (%) | Energy (mJ) | BFSR (%) | Offload x |
|---|---|---|---|---|---|---|
| td3_g0 | 70 | 3 | 9.30±0.52 | 10.77±0.54 | 71.43±13.31 | 0.46±0.01 |

## penalty_form

| Variant | N | Pairs | Failure (%) | Slot P95 (%) | Energy (mJ) | BFSR (%) | Device P95 (%) | Final lambda | d Energy vs sfr_hinge_d0.1 | d Failure vs sfr_hinge_d0.1 |
|---|---|---|---|---|---|---|---|---|---|---|
| sfr_hinge_d0.1 | 70 | 3 | 9.42±0.09 | 9.95±0.07 | 8.41±0.33 | 95.78±1.72 | 27.45±5.88 | 2.95±0.10 | -- | -- |

## ap

| Variant | N | Pairs | Failure (%) | Energy (mJ) | Best AP (%) | AP-load CV | d Energy vs sfr | d Failure vs sfr |
|---|---|---|---|---|---|---|---|---|
| sfr | 70 | 3 | 9.42±0.09 | 8.41±0.33 | 83.00±11.45 | 0.29±0.15 | -- | -- |
| td3 | 70 | 3 | 9.30±0.52 | 10.77±0.54 | 79.99±1.06 | 0.19±0.03 | +2.36 [+1.47, +3.25] | -0.12 [-1.48, +1.23] |

Completed cases: 73 of 217.

