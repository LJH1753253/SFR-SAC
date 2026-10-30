# SFR-SAC

Service-Failure-Regulated Soft Actor-Critic for energy-efficient task
offloading in heterogeneous IIoT-MEC systems.

This repository contains the final v2 implementation used for the Evidence5
study. The proposed policy is internally retained under the compatibility
identifier `safe` and the selected configuration is `split_exact`; the paper
name for that policy is **SFR-SAC**. The implementation uses a split service-
flow actor, structured reward/cost critics, hybrid continuous--categorical
actions, and straight-through AP execution.

The controller uses the immediate slot service-failure cost with `gamma=0`.

## Quick Start

Python 3.10+ and PyTorch are required. Install a CUDA-enabled PyTorch build if
GPU training is desired.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q tests
```

Run a short standalone CPU check with real gradient updates:

```bash
python scripts/train_one.py --method sfr_sac --N 6 --seed 0 \
  --episodes 2 --eval-episodes 2 --horizon 12 --device cpu \
  --output runs/smoke_sfr_sac
```

Evaluate the shipped N=70 example model:

```bash
python scripts/evaluate_model.py --episodes 100 --device cpu \
  --output runs/example_evaluation
```

The example checkpoint is a final N=70, seed-0 SFR-SAC model. Its metadata and
configuration are stored beside `model.pt`.

## Experiment Suite

The reproducible multi-GPU runner is retained in `run_original.sh`. It uses
the physical v2 environment, five paired deployments for the main comparison,
and resume-safe per-run outputs.

```bash
# Linux or WSL
export PYTHON=python
bash run_original.sh smoke
bash run_original.sh dry-run
bash run_original.sh start --hours 24 --workers 2 --gpus 0
```

For a full two-GPU run on a server:

```bash
bash run_original.sh start --hours 46 --workers 8 --gpus 0,1
bash run_original.sh status
tail -n 50 outputs/original_ch5/nohup.log
```

Use `bash run_original.sh stop` for a graceful stop and repeat the same start
command to resume completed and interrupted runs. The controller keeps each
run's configuration, status, log, model, evaluation summary, and checkpoint
separate. Do not overwrite an existing output directory with a different
configuration.

The main grid is `N={50,55,60,65,70}` with paired seeds/layouts. The reported
ablations cover the service-failure budget, service-window fraction, dual
update granularity, fixed penalty, and MEC background load. The exact values
are in `configs/base.json` and the stage manifests produced by the runner.

## Final Results

The final five-pair results are included under `results/final/`:

- `figures/`: publication-ready PDF figures;
- `tables/`: matching CSV and LaTeX tables;
- `data/evidence5_summary.csv`: means and sample standard deviations;
- `data/paired_comparisons.csv`: paired deltas and nominal intervals;
- `data/comparison_locked.json`: validation-only primary-policy selection;
- `data/evidence5_REPORT.md`: complete experiment report.

Energy is reported as mJ/device/slot. Cost, BFSR, and failure components are
reported as percentages. The shipped figures use the terminology
local-computing failure, edge-delivery failure, and MEC-service failure.

## Repository Layout

```text
agents/             actor, critics, action projection, and baselines
envs/               one-slot heterogeneous IIoT-MEC environment
v2/                 training/evaluation runtime and physical environment
configs/            reproducible experiment configuration
original_ch5/       resume-safe main/ablation controller
scripts/            one-run training and checkpoint evaluation entry points
models/             small example checkpoint and provenance metadata
results/final/      final PDFs, CSV summaries, and report
tests/              smoke and protocol tests
```

