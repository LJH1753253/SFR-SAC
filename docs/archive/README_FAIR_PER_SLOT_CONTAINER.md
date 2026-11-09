# Fair Critics + Per-slot Container Run

This package runs only the two revision stages that were not executed in the previous container:

1. `fair_critics`: five paired deployments, including SFR-SAC, Safe TD3 controls, structured-critic controls, and the declared fixed-penalty arms;
2. `per_slot`: Actor, critic refinement from the actor, critic refinement from a neutral start, and the full-information numerical optimizer.

The `per_slot` stage reuses the completed SFR-SAC parent models from `fair_critics`. No historical checkpoints are included in this package.

## Environment

Use the container environment that already has PyTorch, Gymnasium, and pytest. If `/opt/conda/bin/python` does not contain Gymnasium, point `PYTHON` to the container's existing project environment.

```bash
export PYTHON=/opt/conda/bin/python
"$PYTHON" -c 'import torch,gymnasium; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
nvidia-smi
```

The default settings use both GPUs, four GPU workers, and twenty CPU workers:

```bash
export GPUS=0,1
export GPU_WORKERS=4
export CPU_WORKERS=20
export TOTAL_HOURS=31
```

## Smoke and plan

```bash
bash run_fair_per_slot.sh smoke
bash run_fair_per_slot.sh dry-run
```

Smoke outputs are not paper results. The dry-run prints the number of training and evaluation tasks and the estimated resource cost.

## Formal background run

```bash
export PYTHON=/opt/conda/bin/python
export FAIR_PER_SLOT_OUTPUT=$PWD/outputs/fair_per_slot_20261109
export GPUS=0,1
export GPU_WORKERS=4
export CPU_WORKERS=20
export TOTAL_HOURS=31

bash run_fair_per_slot.sh start
```

The controller starts `fair_critics` first and starts `per_slot` only after the first stage finishes. The total wall-clock budget is shared by both stages. The output directory is independent of all earlier runs.

## Monitor and resume

```bash
bash run_fair_per_slot.sh status
tail -f outputs/fair_per_slot_20261109/controller.log
nvidia-smi
```

If the SSH session disconnects, reconnect to the same directory, export the same variables, and run `bash run_fair_per_slot.sh start` again. Completed tasks are skipped by the revision runner.

To stop gracefully:

```bash
bash run_fair_per_slot.sh stop
```

After completion:

```bash
bash run_fair_per_slot.sh report
```

The final report is written to `outputs/fair_per_slot_20261109/report/REVISION_REPORT.md`.
