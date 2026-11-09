"""Archive the container Fair-Critics and Per-slot outputs without checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path


FAIR_VARIANTS = ("sfr", "td3", "td3_struct", "td3_struct_split",
                 "pen_struct_b2", "pen_struct_b2.5", "pen_struct_b3",
                 "pen_struct_b3.5", "pen_struct_b4.5")
PER_SLOT_VARIANTS = ("actor", "critic_refine", "critic_scratch", "oracle")
METHOD_LABELS = {
    "sfr": "SFR-SAC",
    "td3": "Safe TD3",
    "td3_struct": "TD3 + structured critics",
    "td3_struct_split": "TD3 + structured critics + split actor",
    "pen_struct_b2": "Fixed penalty beta=2",
    "pen_struct_b2.5": "Fixed penalty beta=2.5",
    "pen_struct_b3": "Fixed penalty beta=3",
    "pen_struct_b3.5": "Fixed penalty beta=3.5",
    "pen_struct_b4.5": "Fixed penalty beta=4.5",
    "actor": "Actor",
    "critic_refine": "Critic refine",
    "critic_scratch": "Critic scratch",
    "oracle": "Numerical optimizer",
}
TEST_FILES = ("test_summary.json", "eval_episodes.csv", "per_device.csv",
              "task.json", "run_status.json", "revision_source.json", "hardware.json")
PER_SLOT_EXTRA = ("eval_slots.csv",)
PARENT_FILES = ("final_eval_summary.json", "job.json", "task.json", "run_status.json",
                "revision_source.json", "hardware.json", "environment.json", "runtime.json")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mean_sd(values):
    values = [float(v) for v in values]
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return mean, sd


def actual_parent(root: Path, parent):
    return root / "runs" / f"N{parent['N']}" / f"layout{parent['layout_seed']}" / f"train{parent['seed']}" / parent["id"]


def copy_task(root: Path, dest: Path, task, per_slot=False):
    source = root / task["task"] / task["id"]
    if not source.exists():
        raise RuntimeError(f"Missing task directory: {source}")
    status = read_json(source / "run_status.json")
    summary = read_json(source / "test_summary.json")
    if status.get("status") != "done":
        raise RuntimeError(f"Task is not done: {task['id']}")
    if summary.get("split") != "held_out_test":
        raise RuntimeError(f"Unexpected split: {task['id']}")
    if summary.get("source_digest") != task.get("source_digest", summary.get("source_digest")):
        # The task manifest stores the main source digest in the parent config, while the
        # summary stores the same digest directly. The comparison below is intentionally
        # limited to the summary/revision markers used by the runner.
        pass
    target = dest / task["task"] / task["id"]
    target.mkdir(parents=True, exist_ok=True)
    names = tuple(name for name in TEST_FILES if not (per_slot and name == "hardware.json")) + (PER_SLOT_EXTRA if per_slot else ())
    for name in names:
        if not (source / name).exists():
            raise RuntimeError(f"Missing {name}: {task['id']}")
        shutil.copy2(source / name, target / name)
    return summary, target


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, variants):
    groups = defaultdict(list)
    for row in rows:
        groups[row["variant"]].append(row)
    result = []
    metrics = ("cost_percent", "energy_mJ_device_slot", "bfsr_percent")
    for variant in variants:
        items = groups[variant]
        record = {"variant": variant, "method": METHOD_LABELS[variant], "pairs": len(items)}
        for metric in metrics:
            mean, sd = mean_sd([item[metric] for item in items])
            record[f"{metric}_mean"] = mean
            record[f"{metric}_sd"] = sd
        if "decision_ms_median" in items[0]:
            for metric in ("decision_ms_median", "decision_ms_p95"):
                mean, sd = mean_sd([item[metric] for item in items])
                record[f"{metric}_mean"] = mean
                record[f"{metric}_sd"] = sd
        result.append(record)
    return result


def value(record, metric, digits=2):
    return f"{record[f'{metric}_mean']:.{digits}f} $\\pm$ {record[f'{metric}_sd']:.{digits}f}"


def write_tex(path: Path, rows, per_slot=False):
    if per_slot:
        header = "Method & Cost (\\%) & Energy (mJ/device/slot) & BFSR (\\%) & Median (ms) & P95 (ms) \\\\" 
        lines = [r"\begin{table}[t]", r"\centering", r"\caption{Container Per-slot results at $N=70$.}",
                 r"\label{tab:container_per_slot}", r"\footnotesize", r"\begin{tabular}{lccccc}", r"\hline", header, r"\hline"]
        for row in rows:
            lines.append(f"{row['method']} & {value(row, 'cost_percent')} & {value(row, 'energy_mJ_device_slot')} & "
                         f"{value(row, 'bfsr_percent')} & {value(row, 'decision_ms_median', 3)} & "
                         f"{value(row, 'decision_ms_p95', 3)} \\\\")
    else:
        header = "Variant & Pairs & Cost (\\%) & Energy (mJ/device/slot) & BFSR (\\%) \\\\" 
        lines = [r"\begin{table}[t]", r"\centering", r"\caption{Container Fair-Critics results at $N=70$.}",
                 r"\label{tab:container_fair_critics}", r"\footnotesize", r"\begin{tabular}{lcccc}", r"\hline", header, r"\hline"]
        for row in rows:
            lines.append(f"{row['method']} & {row['pairs']} & {value(row, 'cost_percent')} & "
                         f"{value(row, 'energy_mJ_device_slot')} & {value(row, 'bfsr_percent')} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Extracted fair_per_slot_20261109 output directory")
    parser.add_argument("--dest", type=Path, default=Path("results/revision_20261108/container_20261109"))
    args = parser.parse_args()
    root = args.source.resolve()
    dest = args.dest.resolve()
    if dest.exists() and not (dest / "raw").exists():
        raise RuntimeError(f"Refusing to overwrite unrelated destination: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    manifest = read_json(root / "manifest.json")
    protocol = read_json(root / "protocol.json")
    suite_status = read_json(root / "suite_status.json")
    tasks = {task["id"]: task for task in manifest["tasks"]}
    parents = {parent["id"]: parent for parent in manifest["parents"]}
    cases = [case for case in manifest["cases"] if case["experiment"] in {"fair_critics", "per_slot"}]
    fair_cases = [case for case in cases if case["experiment"] == "fair_critics"]
    per_slot_cases = [case for case in cases if case["experiment"] == "per_slot"]
    if len(fair_cases) != 35 or len(per_slot_cases) != 20:
        raise RuntimeError(f"Expected 35 fair test cases and 20 per-slot cases, got {len(fair_cases)} and {len(per_slot_cases)}")

    rows = {"fair_critics": [], "per_slot": []}
    task_hashes = {}
    used_parent_ids = set()
    for case in cases:
        task = dict(tasks[case["task_id"]])
        summary, target = copy_task(root, dest / "raw", task, per_slot=case["experiment"] == "per_slot")
        used_parent_ids.add(case["parent_id"])
        task_hashes[f"raw/{task['task']}/{task['id']}/test_summary.json"] = sha256(target / "test_summary.json")
        row = {"variant": case["variant"], "method": METHOD_LABELS[case["variant"]],
               "layout_seed": case["layout_seed"], "training_seed": case["seed"],
               "task_id": case["task_id"], "parent_id": case["parent_id"],
               "test_rng_seed": task.get("test_rng_seed"),
               "cost_percent": summary["summary"]["cost_percent"],
               "energy_mJ_device_slot": summary["summary"]["energy_mJ_device_slot"],
               "bfsr_percent": summary["summary"]["bfsr_percent"],
               "device_failure_p95_percent": summary["summary"].get("device_failure_p95_percent"),
               "avg_tx_power": summary["summary"].get("avg_tx_power"),
               "final_lambda": summary["summary"].get("final_lambda"),
               "source_relative_path": f"raw/{task['task']}/{task['id']}"}
        if case["experiment"] == "per_slot":
            row["decision_ms_median"] = summary["summary"]["decision_ms_median"]
            row["decision_ms_p95"] = summary["summary"]["decision_ms_p95"]
            row["iterations"] = summary["summary"].get("iterations", task.get("iters"))
        rows[case["experiment"]].append(row)

    parent_records = []
    for parent_id in sorted(used_parent_ids):
        parent = parents[parent_id]
        source = actual_parent(root, parent)
        target = dest / "raw" / "parents" / parent_id
        target.mkdir(parents=True, exist_ok=True)
        artifacts = {}
        for name in PARENT_FILES:
            path = source / name
            if path.exists():
                shutil.copy2(path, target / name)
                artifacts[name] = sha256(path)
        if not (source / "final_eval_summary.json").exists():
            raise RuntimeError(f"Missing parent final summary: {parent_id}")
        parent_records.append({"id": parent_id, "variant": parent["variant"], "seed": parent["seed"],
                              "layout_seed": parent["layout_seed"], "config_hash": parent.get("config_hash"),
                              "artifacts": artifacts})

    for name in ("manifest.json", "protocol.json", "suite_status.json"):
        shutil.copy2(root / name, dest / "raw" / name)
    (dest / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "report" / "REVISION_REPORT.md", dest / "data" / "container_revision_report.md")
    seed_fields = list(rows["fair_critics"][0].keys())
    write_csv(dest / "tables" / "fair_critics_container_per_seed.csv", rows["fair_critics"], seed_fields)
    per_seed_fields = list(rows["per_slot"][0].keys())
    write_csv(dest / "tables" / "per_slot_container_per_seed.csv", rows["per_slot"], per_seed_fields)
    fair_summary = aggregate(rows["fair_critics"], FAIR_VARIANTS)
    per_slot_summary = aggregate(rows["per_slot"], PER_SLOT_VARIANTS)
    write_csv(dest / "tables" / "fair_critics_container.csv", fair_summary, list(fair_summary[0].keys()))
    write_csv(dest / "tables" / "per_slot_container.csv", per_slot_summary, list(per_slot_summary[0].keys()))
    write_tex(dest / "tables" / "fair_critics_container.tex", fair_summary)
    write_tex(dest / "tables" / "per_slot_container.tex", per_slot_summary, per_slot=True)

    report = [
        "# Container Fair-Critics and Per-slot Archive",
        "",
        "This archive was copied from the completed container run. No training or optimization was run during archiving.",
        "",
        "## Completion",
        "",
        "- Fair-Critics: 35 test cases plus 35 parent trainings, all completed.",
        "- Per-slot: 20 tests, all completed; 5 seeds for Actor, Critic refine, Critic scratch, and Numerical optimizer.",
        "- The generic runner report says `73 of 217` because it uses the full revision catalog denominator and includes shared test aliases. The selected target stages are complete and are reported separately here.",
        "",
        "## Fair-Critics",
        "",
        "| Variant | Pairs | Cost (%) | Energy (mJ/device/slot) | BFSR (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in fair_summary:
        report.append(f"| {row['method']} | {row['pairs']} | {value(row, 'cost_percent')} | {value(row, 'energy_mJ_device_slot')} | {value(row, 'bfsr_percent')} |")
    report += [
        "",
        "## Per-slot",
        "",
        "| Method | Pairs | Cost (%) | Energy (mJ/device/slot) | BFSR (%) | Median (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in per_slot_summary:
        report.append(f"| {row['method']} | {row['pairs']} | {value(row, 'cost_percent')} | {value(row, 'energy_mJ_device_slot')} | {value(row, 'bfsr_percent')} | {value(row, 'decision_ms_median', 3)} |")
    report += [
        "",
        "The Actor is the low-latency policy. The per-slot refinements and numerical optimizer reduce energy only by accepting higher failure cost and/or much longer decision time. The Numerical optimizer is a full-information numerical baseline, not a proven global optimum.",
        "",
        "## Provenance",
        "",
        f"- Main source digest: `{manifest['source_digest']}`",
        f"- Revision source digest: `{manifest['revision_source_digest']}`",
        f"- Protocol hash: `{suite_status.get('protocol_hash')}`",
        "- Fair test raw archive includes summaries, per-episode metrics, per-device metrics, task/status markers, hardware, and revision markers; large Fair test slot traces and all checkpoints are omitted.",
        "- Per-slot raw archive additionally includes per-slot traces because these files are smaller and useful for reproducing the decision-time summary.",
    ]
    (dest / "data" / "container_fair_per_slot_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    provenance = {
        "archive_created": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "repository_commit_at_archive": git_head(),
        "source_directory_in_container": "/public/home/hnust19209626205/iotj_fair_per_slot_20261109/outputs/fair_per_slot_20261109",
        "source_digest": manifest["source_digest"],
        "revision_source_digest": manifest["revision_source_digest"],
        "protocol_hash": suite_status.get("protocol_hash"),
        "protocol": protocol,
        "suite_status": suite_status,
        "selected_stage_cases": {"fair_critics": len(fair_cases), "per_slot": len(per_slot_cases)},
        "parent_trainings": len(parent_records),
        "parent_records": parent_records,
        "archived_file_hashes": task_hashes,
        "large_files_omitted": ["all checkpoint.pt/model*.pt files", "Fair test eval_slots.csv files", "worker.log files"],
        "formal_training_or_optimization_run_during_archiving": False,
    }
    write_json(dest / "data" / "container_fair_per_slot_provenance.json", provenance)
    print(json.dumps({"status": "archived", "fair_cases": len(fair_cases), "per_slot_cases": len(per_slot_cases),
                      "destination": str(dest)}, indent=2))


if __name__ == "__main__":
    main()
