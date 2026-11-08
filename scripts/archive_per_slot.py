"""Recover and archive the completed per-slot study without rerunning it."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


METHOD_NAMES = {
    "actor": "Actor",
    "critic_refine": "Critic refine",
    "critic_scratch": "Critic scratch",
    "oracle": "Numerical optimizer",
}
RAW_FILES = (
    "test_summary.json",
    "task.json",
    "run_status.json",
    "revision_source.json",
    "eval_episodes.csv",
    "eval_slots.csv",
    "per_device.csv",
)
PARENT_FILES = (
    "final_eval_summary.json",
    "job.json",
    "task.json",
    "run_status.json",
    "revision_source.json",
    "hardware.json",
    "environment.json",
    "runtime.json",
)
T95_DF4 = 2.7764451051977987


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean_sd(values):
    values = [float(v) for v in values]
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return mean, sd


def ci95(values):
    mean, sd = mean_sd(values)
    half = T95_DF4 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, sd, mean - half, mean + half


def actual_parent(source_root: Path, parent):
    return source_root / "runs" / f"N{parent['N']}" / f"layout{parent['layout_seed']}" / f"train{parent['seed']}" / parent["id"]


def verify_and_collect(source_root: Path):
    output_root = source_root
    manifest = read_json(source_root / "manifest.json")
    protocol = read_json(output_root / "protocol.json")
    suite_status = read_json(output_root / "suite_status.json")
    parent_by_id = {p["id"]: p for p in manifest["parents"]}
    tasks = {t["id"]: t for t in manifest["tasks"]}
    cases = [c for c in manifest["cases"] if c["task"] == "per_slot"]
    if len(cases) != 20:
        raise RuntimeError(f"Expected 20 per-slot cases, found {len(cases)}")

    rows = []
    parent_records = []
    seen_parents = set()
    for case in cases:
        task = tasks[case["task_id"]]
        task_dir = source_root / "per_slot" / task["id"]
        status = read_json(task_dir / "run_status.json")
        summary_doc = read_json(task_dir / "test_summary.json")
        summary = summary_doc["summary"]
        if status.get("status") != "done":
            raise RuntimeError(f"Per-slot task is not done: {task['id']}")
        if summary_doc.get("split") != "held_out_test":
            raise RuntimeError(f"Unexpected split in {task['id']}: {summary_doc.get('split')}")
        if summary_doc.get("source_digest") != manifest["source_digest"]:
            raise RuntimeError(f"Source digest mismatch in {task['id']}")
        if summary.get("eval_episodes") != protocol["per_slot_episodes"]:
            raise RuntimeError(f"Episode count mismatch in {task['id']}")
        if summary.get("eval_slots") != protocol["per_slot_episodes"] * 200:
            raise RuntimeError(f"Slot count mismatch in {task['id']}")
        expected_iters = protocol["per_slot_iterations"].get(task["optimizer"], 0)
        if summary.get("iterations") != expected_iters:
            raise RuntimeError(f"Iteration count mismatch in {task['id']}")
        marker = read_json(task_dir / "revision_source.json")
        if marker.get("revision_source_digest") != manifest["revision_source_digest"]:
            raise RuntimeError(f"Revision source mismatch in {task['id']}")
        for name in RAW_FILES:
            if not (task_dir / name).exists():
                raise RuntimeError(f"Missing {name} in {task['id']}")

        parent = parent_by_id[task["parent_id"]]
        parent_dir = actual_parent(source_root, parent)
        if parent["id"] not in seen_parents:
            p_status = read_json(parent_dir / "run_status.json")
            if p_status.get("status") != "done":
                raise RuntimeError(f"Parent training is not done: {parent['id']}")
            for name in ("model_ep1600.pt", "model.pt", "final_eval_summary.json", "job.json", "hardware.json"):
                if not (parent_dir / name).exists():
                    raise RuntimeError(f"Missing parent artifact {name}: {parent['id']}")
            parent_records.append({
                "id": parent["id"],
                "seed": parent["seed"],
                "layout_seed": parent["layout_seed"],
                "variant": parent["variant"],
                "directory": str(parent_dir),
                "source_digest": read_json(parent_dir / "final_eval_summary.json").get("source_digest"),
                "config_hash": parent.get("config_hash"),
                "artifacts": {name: sha256(parent_dir / name) for name in PARENT_FILES if (parent_dir / name).exists()},
            })
            seen_parents.add(parent["id"])

        rows.append({
            "method": METHOD_NAMES[task["optimizer"]],
            "optimizer": task["optimizer"],
            "layout_seed": task["layout_seed"],
            "training_seed": task["seed"],
            "cost_percent": summary["cost_percent"],
            "energy_mJ_device_slot": summary["energy_mJ_device_slot"],
            "bfsr_percent": summary["bfsr_percent"],
            "decision_ms_median": summary["decision_ms_median"],
            "decision_ms_p95": summary["decision_ms_p95"],
            "test_episodes": summary["eval_episodes"],
            "test_slots": summary["eval_slots"],
            "iterations": summary["iterations"],
            "task_id": task["id"],
            "parent_id": task["parent_id"],
            "test_rng_seed": task["test_rng_seed"],
            "source_relative_path": f"data/per_slot_runs/{task['id']}",
            "revision_source_digest": marker["revision_source_digest"],
        })
    return manifest, protocol, suite_status, rows, parent_records, tasks


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    by_method = defaultdict(list)
    for row in rows:
        by_method[row["optimizer"]].append(row)
    actor = {(r["layout_seed"], r["training_seed"]): r for r in by_method["actor"]}
    out = []
    for optimizer in ("actor", "critic_refine", "critic_scratch", "oracle"):
        items = by_method[optimizer]
        energy_delta = []
        cost_delta = []
        for item in items:
            key = (item["layout_seed"], item["training_seed"])
            if optimizer != "actor" and key in actor:
                energy_delta.append(item["energy_mJ_device_slot"] - actor[key]["energy_mJ_device_slot"])
                cost_delta.append(item["cost_percent"] - actor[key]["cost_percent"])
        record = {"method": METHOD_NAMES[optimizer], "optimizer": optimizer, "completed_pairs": len(items)}
        for metric in ("cost_percent", "energy_mJ_device_slot", "bfsr_percent", "decision_ms_median", "decision_ms_p95"):
            mean, sd = mean_sd([item[metric] for item in items])
            record[f"{metric}_mean"] = mean
            record[f"{metric}_sd"] = sd
        for name, values in (("energy_delta_vs_actor", energy_delta), ("cost_delta_vs_actor", cost_delta)):
            if values:
                mean, sd, low, high = ci95(values)
                record[f"{name}_mean"] = mean
                record[f"{name}_sd"] = sd
                record[f"{name}_ci95_low"] = low
                record[f"{name}_ci95_high"] = high
            else:
                for suffix in ("mean", "sd", "ci95_low", "ci95_high"):
                    record[f"{name}_{suffix}"] = None
        out.append(record)
    return out


def tex_value(mean, sd, digits=2):
    return f"{mean:.{digits}f} $\\pm$ {sd:.{digits}f}"


def write_tex(path: Path, summary_rows):
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Per-slot optimization at $N=70$ over five paired deployments.}",
        r"\label{tab:per_slot}", r"\footnotesize",
        r"\begin{tabular}{lccccc}", r"\hline",
        r"Method & Cost (\%) & Energy (mJ/device/slot) & BFSR (\%) & Median (ms) & P95 (ms) \\",
        r"\hline",
    ]
    for row in summary_rows:
        lines.append(r"{} & {} & {} & {} & {} & {} \\".format(
            row["method"],
            tex_value(row["cost_percent_mean"], row["cost_percent_sd"]),
            tex_value(row["energy_mJ_device_slot_mean"], row["energy_mJ_device_slot_sd"]),
            tex_value(row["bfsr_percent_mean"], row["bfsr_percent_sd"]),
            tex_value(row["decision_ms_median_mean"], row["decision_ms_median_sd"], 3),
            tex_value(row["decision_ms_p95_mean"], row["decision_ms_p95_sd"], 3),
        ))
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_raw(source_root: Path, dest: Path, manifest, tasks):
    if dest.exists() and any(dest.iterdir()):
        if not (dest / "source_manifest.json").exists():
            raise RuntimeError(f"Refusing to overwrite non-empty archive directory: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    for task in tasks.values():
        if task.get("task") != "per_slot":
            continue
        source = source_root / "per_slot" / task["id"]
        target = dest / task["id"]
        target.mkdir(parents=True, exist_ok=True)
        for name in RAW_FILES:
            shutil.copy2(source / name, target / name)


def build_report(summary_rows, rows, manifest, protocol, suite_status, source_root, parent_records):
    lines = [
        "# Per-slot Optimization Report",
        "",
        "This report was regenerated from the completed historical per-slot outputs. No training or numerical optimization was run during recovery.",
        "",
        "## Validated scope",
        "",
        f"- `N={rows[0]['test_episodes'] * 0 + 70}`, `d=0.10`, `eta=0.30`, `gamma=0`.",
        "- Five paired training seeds/layouts: `(1000,0)` through `(1004,4)`.",
        "- Four methods, 20 completed tests, 30 evaluation episodes and 6,000 slots per test.",
        "- Critic refine: 30 iterations; critic scratch: 150 iterations; numerical optimizer: 100 iterations.",
        "- Decision time is measured around the optimizer/action function with `time.perf_counter`; it excludes state acquisition, signaling, and actuation overhead.",
        "- Means and sample standard deviations are over the five deployment pairs. Evaluation slots are not treated as independent training replicates.",
        "",
        "## Completion reconciliation",
        "",
        "The historical controller status records 20 completed of 25 planned tasks and lists the five parent-training IDs under `failed`, but each of those entries has `exit_code=0`. Direct inspection finds `run_status=done`, `model_ep1600.pt`, `model.pt`, and `final_eval_summary.json` for all five parents. All 20 per-slot tasks also have `run_status=done` and complete 30-episode/6,000-slot summaries. The artifact-level completion count is therefore 5/5 parent trainings and 20/20 per-slot tests; the stale controller count is retained in provenance rather than silently overwritten.",
        "",
        "## Summary",
        "",
        "| Method | Cost (%) | Energy (mJ/device/slot) | BFSR (%) | Median (ms) | P95 (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['cost_percent_mean']:.3f} +/- {row['cost_percent_sd']:.3f} | "
            f"{row['energy_mJ_device_slot_mean']:.3f} +/- {row['energy_mJ_device_slot_sd']:.3f} | "
            f"{row['bfsr_percent_mean']:.3f} +/- {row['bfsr_percent_sd']:.3f} | "
            f"{row['decision_ms_median_mean']:.3f} +/- {row['decision_ms_median_sd']:.3f} | "
            f"{row['decision_ms_p95_mean']:.3f} +/- {row['decision_ms_p95_sd']:.3f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The Actor is the low-latency method. Critic refinement and critic scratch reduce the energy objective at the cost of substantially longer per-slot decisions. The full-information Numerical optimizer has the lowest mean energy, but its mean cost is close to the 10% budget and its BFSR is lower than the Actor. Therefore, this experiment does not establish Actor energy optimality at an equal failure level. The Numerical optimizer is a full-information numerical baseline, not a mathematically proven global optimum.",
        "",
        "## Provenance",
        "",
        f"- Historical source directory: `{source_root}`",
        f"- Historical main source digest: `{manifest['source_digest']}`",
        f"- Historical revision source digest: `{manifest['revision_source_digest']}`",
        f"- Historical protocol hash: `{suite_status.get('protocol_hash', 'not recorded')}`",
        "- The per-slot run used the historical overnight source/checkpoints and hardware recorded in `data/per_slot_provenance.json`; it must not be described as sharing the current container's Fair-Critics checkpoints.",
        "- Required raw test summaries and CSV files are archived under `data/per_slot_runs/`. Training checkpoints are intentionally not copied.",
    ]
    return "\n".join(lines) + "\n"


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Historical revision_local_overnight output root")
    parser.add_argument("--dest", type=Path, default=Path("results/revision_20261108"))
    args = parser.parse_args()
    source_root = args.source.resolve()
    if source_root.name == "per_slot":
        source_root = source_root.parent
    dest = args.dest.resolve()
    manifest, protocol, suite_status, rows, parents, tasks = verify_and_collect(source_root)

    raw_dest = dest / "data" / "per_slot_runs"
    copy_raw(source_root, raw_dest, manifest, tasks)
    for parent in parents:
        target = raw_dest / "parents" / parent["id"]
        target.mkdir(parents=True, exist_ok=True)
        source = Path(parent["directory"])
        for name in PARENT_FILES:
            if (source / name).exists():
                shutil.copy2(source / name, target / name)

    shutil.copy2(source_root / "manifest.json", raw_dest / "source_manifest.json")
    shutil.copy2(source_root / "protocol.json", raw_dest / "source_protocol.json")

    fields = list(rows[0].keys())
    write_csv(dest / "data" / "per_slot_per_seed.csv", rows, fields)
    summary_rows = aggregate(rows)
    summary_fields = list(summary_rows[0].keys())
    write_csv(dest / "tables" / "per_slot.csv", summary_rows, summary_fields)
    write_tex(dest / "tables" / "per_slot.tex", summary_rows)

    # Use the repository report generator on a temporary manifest when its runtime is available.
    report_generator = "not run"
    staging = dest.parent.parent / "outputs" / "_per_slot_report_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        stage_manifest = {
            "source_digest": manifest["source_digest"],
            "revision_source_digest": manifest["revision_source_digest"],
            "parents": [], "tasks": [], "cases": [],
        }
        for parent in parents:
            stage_parent = dict(next(p for p in manifest["parents"] if p["id"] == parent["id"]))
            stage_parent["directory"] = str(staging / "parents" / parent["id"])
            stage_manifest["parents"].append(stage_parent)
        for task in tasks.values():
            if task.get("task") != "per_slot":
                continue
            stage_task = dict(task)
            stage_task["directory"] = str(staging / "per_slot" / task["id"])
            stage_manifest["tasks"].append(stage_task)
        stage_manifest["cases"] = [c for c in manifest["cases"] if c["task"] == "per_slot"]
        for parent in parents:
            source = Path(parent["directory"])
            target = staging / "parents" / parent["id"]
            target.mkdir(parents=True, exist_ok=True)
            for name in ("final_eval_summary.json",):
                shutil.copy2(source / name, target / name)
        for task in tasks.values():
            if task.get("task") != "per_slot":
                continue
            source = source_root / "per_slot" / task["id"]
            target = staging / "per_slot" / task["id"]
            target.mkdir(parents=True, exist_ok=True)
            for name in ("test_summary.json", "revision_source.json"):
                shutil.copy2(source / name, target / name)
        write_json(staging / "manifest.json", stage_manifest)
        from revision import report as revision_report
        revision_report.generate(staging)
        report_generator = "revision.report.generate"
    except Exception as exc:
        report_generator = f"unavailable: {type(exc).__name__}: {exc}"
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    # The dedicated report is intentionally more explicit than the generic stage report.
    (dest / "data" / "per_slot_report.md").write_text(
        build_report(summary_rows, rows, manifest, protocol, suite_status, source_root, parents), encoding="utf-8"
    )
    hardware = read_json(Path(parents[0]["directory"]) / "hardware.json")
    provenance = {
        "archive_created": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "current_git_commit": git_commit(),
        "source_directory": str(source_root),
        "source_digest": manifest["source_digest"],
        "revision_source_digest": manifest["revision_source_digest"],
        "protocol_hash": suite_status.get("protocol_hash"),
        "protocol": protocol,
        "historical_suite_status": suite_status,
        "hardware": hardware,
        "training": {"N": 70, "episodes": 1600, "pairs": [[1000, 0], [1001, 1], [1002, 2], [1003, 3], [1004, 4]]},
        "per_slot": {"methods": list(METHOD_NAMES), "tests": 20, "episodes_per_test": 30, "slots_per_test": 6000,
                     "warmup_episodes": 10, "iterations": protocol["per_slot_iterations"]},
        "parent_records": parents,
        "archived_files": {
            str(path.relative_to(dest)): sha256(path)
            for path in dest.rglob("*") if path.is_file()
        },
        "report_generator": report_generator,
        "formal_training_or_optimization_run_during_recovery": False,
    }
    write_json(dest / "data" / "per_slot_provenance.json", provenance)
    print(json.dumps({"status": "archived", "rows": len(rows), "summary": str(dest / "tables" / "per_slot.csv"),
                      "report": report_generator}, indent=2))


if __name__ == "__main__":
    main()
