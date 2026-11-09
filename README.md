# SFR-SAC

Service-Failure-Regulated Soft Actor-Critic for energy-efficient task offloading in heterogeneous IIoT-MEC systems.

## Research Problem

The project studies joint task offloading, transmit-power control, and AP association under an aggregate service-failure budget. Workloads are completed within each service slot. Unfinished local-computing, edge-delivery, and MEC-service workload contributes to the service-failure cost.

## Method

SFR-SAC uses a split service-flow actor, structured reward and cost critics, continuous offloading/power actions, and a categorical AP-association branch with straight-through execution. The physical energy model uses CPU-cycle local energy and PA/circuit uplink energy. The default implementation predicts immediate slot outcomes with `gamma=0`; the repository does not claim a hard long-horizon safety guarantee.

## Final Results

The canonical Paper1 results are under [`results/paper/`](results/paper/):

- [`REPORT.md`](results/paper/REPORT.md): complete experimental report;
- [`PROVENANCE.md`](results/paper/PROVENANCE.md): source and protocol mapping;
- [`data_sources.json`](results/paper/data_sources.json): machine-readable table/figure sources;
- [`figures/`](results/paper/figures/): final PDF and PNG figures;
- [`tables/`](results/paper/tables/): canonical CSV and LaTeX tables;
- [`per_seed/`](results/paper/per_seed/): paired per-seed evidence.

At `N=70`, the canonical container results report SFR-SAC at `9.38 +/- 0.16%` service-failure cost, `8.59 +/- 0.36 mJ/device/slot`, and `96.23 +/- 2.31%` BFSR. The Per-slot Actor result is `9.38 +/- 0.16%`, `8.59 +/- 0.38 mJ/device/slot`, `96.37 +/- 2.12%` BFSR, and `1.750 +/- 0.013 ms` median decision time.

## Reproduction

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
python scripts/plot_paper_results.py
python scripts/validate_paper_results.py
```

For a short CPU smoke run:

```bash
python scripts/train_one.py --method sfr_sac --N 6 --seed 0 \
  --episodes 2 --eval-episodes 2 --horizon 12 --device cpu \
  --output runs/smoke_sfr_sac
```

The multi-GPU training and revision runners remain available for reproducing the recorded protocols:

```bash
bash run_original.sh smoke
bash run_revision.sh smoke
```

The full result package does not include large checkpoints or replay buffers. Historical run directories remain in the repository history and under `results/revision_20261108/` for provenance; they are not the primary reader entry point.

## Limitations

- Aggregate cost and BFSR do not guarantee per-device reliability.
- The AP-independent ablation becomes degenerate when all candidate AP gains are symmetric.
- Numerical and model-based baselines may use simulator information unavailable to the learned actor.
- Reported decision latency excludes state acquisition, signaling, and actuation.
- Some revision controls use three paired deployments; the canonical Fair-Critics and Per-slot container rerun uses five pairs.

## Repository Layout

```text
agents/             actor, critics, action projection, and baselines
envs/               one-slot heterogeneous IIoT-MEC environment
v2/                 physical environment and training/evaluation runtime
configs/            reproducible configurations
revision/           revision-study runners and stage definitions
scripts/            training, evaluation, validation, and plotting tools
results/paper/      canonical Paper1 figures, tables, report, and provenance
results/final/      original formal main-study export
results/revision_20261108/  revision-stage source archives
tests/              smoke and protocol tests
```

## Citation

Citation information will be added after publication. No DOI, volume, issue, or acceptance status is claimed in this repository.
