"""Validate the canonical Paper1 result package without running experiments."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "results" / "paper"


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def mean(values):
    return sum(values) / len(values)


def check(condition, message, failures):
    if not condition:
        failures.append(message)


def compare_aggregate(seed_rows, summary_rows, key, metrics, failures):
    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[row[key]].append(row)
    summary_by = {row[key]: row for row in summary_rows}
    for group, items in grouped.items():
        if group not in summary_by:
            failures.append(f"Missing aggregate row: {key}={group}")
            continue
        target = summary_by[group]
        for source_key, summary_key in metrics:
            values = [float(item[source_key]) for item in items]
            actual = mean(values)
            expected = float(target[summary_key])
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-8):
                failures.append(f"Aggregate mismatch {group}/{summary_key}: {actual} != {expected}")


def main():
    failures = []
    required = [
        PAPER / "data_sources.json",
        PAPER / "tables/main_results.csv",
        PAPER / "tables/fair_critics.csv",
        PAPER / "tables/per_slot.csv",
        PAPER / "per_seed/fair_critics_per_seed.csv",
        PAPER / "per_seed/per_slot_per_seed.csv",
    ]
    for path in required:
        check(path.exists(), f"Missing required file: {path}", failures)
    if failures:
        raise SystemExit("\n".join(failures))

    main_rows = read_csv(PAPER / "tables/main_results.csv")
    check(len(main_rows) == 25, "Main table must contain 25 rows", failures)
    check({int(row["N"]) for row in main_rows} == {50, 55, 60, 65, 70}, "Main N set mismatch", failures)
    check({row["Algorithm"] for row in main_rows} == {
        "SFR-SAC", "Penalty SAC", "Safe TD3", "Lyapunov-inspired rule", "Vanilla SAC"
    }, "Main algorithm set mismatch", failures)

    fair_seed = read_csv(PAPER / "per_seed/fair_critics_per_seed.csv")
    fair = read_csv(PAPER / "tables/fair_critics.csv")
    check(len(fair_seed) == 35, "Fair-Critics per-seed row count must be 35", failures)
    check(len([row for row in fair_seed if row["variant"] == "sfr"]) == 5, "Fair SFR seed count must be 5", failures)
    compare_aggregate(
        fair_seed, fair, "variant",
        (("cost_percent", "cost_percent_mean"), ("energy_mJ_device_slot", "energy_mJ_device_slot_mean"),
         ("bfsr_percent", "bfsr_percent_mean")), failures)

    slot_seed = read_csv(PAPER / "per_seed/per_slot_per_seed.csv")
    slot = read_csv(PAPER / "tables/per_slot.csv")
    check(len(slot_seed) == 20, "Per-slot per-seed row count must be 20", failures)
    check({row["variant"] for row in slot_seed} == {"actor", "critic_refine", "critic_scratch", "oracle"},
          "Per-slot method set mismatch", failures)
    compare_aggregate(
        slot_seed, slot, "variant",
        (("cost_percent", "cost_percent_mean"), ("energy_mJ_device_slot", "energy_mJ_device_slot_mean"),
         ("bfsr_percent", "bfsr_percent_mean")), failures)
    provenance = json.loads((ROOT / "results/revision_20261108/container_20261109/data/container_fair_per_slot_provenance.json").read_text(encoding="utf-8"))
    per_slot_protocol = provenance["protocol"]["per_slot_episodes"]
    check(per_slot_protocol == 30 and per_slot_protocol * 200 == 6000,
          "Per-slot protocol must specify 30 episodes and 6000 slots", failures)

    data_sources = json.loads((PAPER / "data_sources.json").read_text(encoding="utf-8"))
    check(data_sources["fair_critics"]["source_digest"].startswith("36c1289d"), "Fair source digest mismatch", failures)
    check(data_sources["per_slot"]["revision_source_digest"].startswith("7e9709f3"), "Per-slot revision digest mismatch", failures)

    changed_core = subprocess.check_output(
        ["git", "diff", "--name-only", "main...HEAD", "--", "agents", "cloud24", "envs", "v2", "revision"],
        text=True,
    ).strip()
    check(not changed_core, f"Core source changed on finalization branch: {changed_core}", failures)

    figures = list((PAPER / "figures").glob("Fig*.pdf"))
    check(len(figures) >= 10, f"Expected at least 10 generated PDF figures, found {len(figures)}", failures)
    if failures:
        raise SystemExit("\n".join(failures))
    print(json.dumps({"status": "passed", "main_rows": len(main_rows), "fair_seed_rows": len(fair_seed),
                      "per_slot_seed_rows": len(slot_seed), "figures": len(figures)}, indent=2))


if __name__ == "__main__":
    main()
