# GPU Container Runbook

## 1. Prepare the Checkout

Upload the revision bundle and the project archive to the server, then verify
the bundle before using it:

```bash
git clone https://github.com/LJH1753253/SFR-SAC.git sfr-sac-main
cd sfr-sac-main
git bundle verify /path/to/sfr-sac-revision-extensions.bundle
git fetch /path/to/sfr-sac-revision-extensions.bundle \
  revision-extensions:refs/remotes/local/revision-extensions
git switch -c revision-extensions refs/remotes/local/revision-extensions
```

Do not run the revision stages from the historical `main` checkout and do not
use the revision checkout to resume historical `original_ch5` results.

## 2. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
nvidia-smi
```

Use the container's CUDA-enabled PyTorch build. Do not replace it with a CPU
wheel.

## 3. Preflight Checks

```bash
python -m py_compile $(find agents envs offloading v2 revision cloud24 -name '*.py')
python -m pytest -q tests/test_revision.py -k 'not new_arms_train_and_reload'
python -m revision.runner --output outputs/revision --dry-run --hours 1
```

The dry-run must show 175 unique trainings and 157 CPU tasks. The first
approved quick rehearsal may run the four tiny training tests; it is not a
result-producing experiment.

## 4. Smoke and Regression

After the static checks pass and before formal training:

```bash
bash run_revision.sh smoke
```

The smoke creates tiny CPU outputs. Keep them separate from the formal output
directory. For a main-grid regression check, create a separate checkout at the
historical main commit and run:

```bash
python -m revision.regression_check ../sfr-sac-main .
```

This is a tiny training check, not a paper experiment. It must not overwrite
historical results.

## 5. Formal Run

Estimate scheduling first:

```bash
bash run_revision.sh dry-run --gpus 0,1 --workers 8 --cpu-workers 8
```

Start only after checking the container lifetime and available disk:

```bash
bash run_revision.sh start --gpus 0,1 --workers 8 --cpu-workers 8 --hours 48
```

The `--hours` value is a wall-clock budget, not a completion guarantee.

## 6. Monitoring and Resume

```bash
bash run_revision.sh status
tail -f outputs/revision/nohup.log
find outputs/revision -name run_status.json | wc -l
```

Stop gracefully at an episode boundary:

```bash
bash run_revision.sh stop
```

Resume the same output directory only with the same branch and source digest:

```bash
bash run_revision.sh start --resume --gpus 0,1 --workers 8 --cpu-workers 8 --hours 24
```

If a task failed for an external resource reason, inspect its `worker.log` and
then use `--retry-failed`. Do not delete successful run directories.

## 7. Stage-By-Stage Execution

```bash
bash run_revision.sh start --stage fair_critics --resume --gpus 0,1 --workers 8
bash run_revision.sh start --stage per_slot --resume --cpu-workers 8
bash run_revision.sh start --stage cpo --resume --gpus 0,1 --workers 8 --cpu-workers 8
```

CPU stages include CPO, MB-greedy, per-slot optimizers, and latency tasks;
GPU stages include neural training.

## 8. Reports and Handoff

After the controller is stopped:

```bash
bash run_revision.sh report
```

Collect `outputs/revision/report/REVISION_REPORT.md`, its CSV/LaTeX tables,
stage manifests, `suite_status.json`, and the per-run summaries, statuses, and
logs.

## 9. Reproducibility Rules

Do not change code, configs, seeds, budgets, test streams, or statistical
thresholds after inspecting results. Do not copy old main checkpoints into the
revision output directory. Preserve the bundle commit, both source digests,
protocol JSON, and raw CSV files with the paper materials.
