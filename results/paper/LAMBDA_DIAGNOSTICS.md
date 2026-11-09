# Lambda Diagnostics

## Recorded result

The Hinge and Linear penalty table reports the following final lambda means at `N=70`:

| Penalty | d=0.08 | d=0.10 | d=0.12 |
|---|---:|---:|---:|
| Hinge | 3.42 +/- 0.11 | 2.95 +/- 0.10 | 2.57 +/- 0.05 |
| Linear | 1.05 +/- 0.55 | 3.29 +/- 2.21 | 5.40 +/- 0.78 |

The Linear values increase with the tested budget, while its cost and energy also have large variability. This pattern is a recorded result, not a value corrected during packaging.

## Code facts

The current implementation sets `cost_limit=d` and selects `penalty_mode` in `revision/variants.py`. In `agents/safe_sac_agent.py`, the actor penalty is:

- Hinge: `lambda * cost_coef * max(Qc-d, 0)`;
- Linear: `lambda * cost_coef * (Qc-d)`.

The batch dual statistic is the mean replay cost (or the selected policy-cost statistic), and the update uses `diff=statistic-d`. Adaptive updates use `lambda_lr*1.5` for positive `diff` and `lambda_lr*0.8` otherwise, followed by clipping to `[0.01,50]`.

## What can be concluded

The formula alone does not predict the observed increase of final Linear lambda with a looser budget. Changing `d` changes the actor objective, the learned policy, the critic predictions, and the replay distribution. The archived result contains aggregate summaries but no retained lambda trajectory logs for these training runs, so the trajectory-level cause cannot be verified.

The defensible paper statement is therefore: the Linear ablation exhibited higher variability and a budget-dependent final-lambda pattern in this experiment; the available logs do not establish a general monotonic law for that pattern. The Hinge form provided the more stable constrained operating point in the recorded comparison.
