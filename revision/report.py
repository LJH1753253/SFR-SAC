"""Tables for the revision study: CSV, LaTeX, and one Markdown report (`report/REVISION_REPORT.md`).

Means and sample standard deviations are over paired deployments. Paired differences against each
stage's reference use nominal two-sided 95 % t intervals (no multiple-comparison correction).
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from revision import variants as V
from revision.source import revision_source_digest
from v2.runtime import read_json

METRICS = ("cost_percent", "energy_mJ_device_slot", "bfsr_percent", "cost_p95_percent",
           "device_failure_p95_percent", "devices_above_budget", "avg_tx_power", "avg_offload_ratio",
           "best_ap_percent", "ap_load_cv", "final_lambda", "decision_ms_median", "decision_ms_p95")
SHORT = dict(cost_percent="Failure (%)", energy_mJ_device_slot="Energy (mJ)", bfsr_percent="BFSR (%)",
             cost_p95_percent="Slot P95 (%)", device_failure_p95_percent="Device P95 (%)",
             devices_above_budget="EDs > d", avg_tx_power="Power (W)", avg_offload_ratio="Offload x",
             best_ap_percent="Best AP (%)", ap_load_cv="AP-load CV", final_lambda="Final lambda",
             decision_ms_median="Decision (ms)", decision_ms_p95="Decision P95 (ms)")
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
STAGE_COLUMNS = {
    "fair_critics": ("cost_percent", "energy_mJ_device_slot", "bfsr_percent", "device_failure_p95_percent",
                     "avg_tx_power", "final_lambda"),
    "per_slot": ("cost_percent", "energy_mJ_device_slot", "bfsr_percent", "decision_ms_median", "decision_ms_p95"),
    "mismatch": ("cost_percent", "energy_mJ_device_slot", "bfsr_percent", "device_failure_p95_percent"),
    "cpo": ("cost_percent", "energy_mJ_device_slot", "bfsr_percent", "final_lambda"),
    "horizon": ("cost_percent", "energy_mJ_device_slot", "bfsr_percent", "avg_offload_ratio"),
    "penalty_form": ("cost_percent", "cost_p95_percent", "energy_mJ_device_slot", "bfsr_percent",
                     "device_failure_p95_percent", "final_lambda"),
    "ap": ("cost_percent", "energy_mJ_device_slot", "best_ap_percent", "ap_load_cv"),
}
REFERENCE = {"fair_critics": "sfr", "per_slot": "actor", "mismatch": "nominal_sfr", "penalty_form": "sfr_hinge_d0.1",
             "ap": "sfr", "horizon": None, "cpo": None}


def mean_sd(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return None, None, 0
    m = sum(values) / len(values)
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return m, sd, len(values)


def load(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    tasks = {t["id"]: t for t in manifest["tasks"]}
    rows = []
    expected_revision = manifest.get("revision_source_digest", revision_source_digest())
    for case in manifest["cases"]:
        task = tasks.get(case["task_id"])
        if task is None:
            continue
        marker = Path(task["directory"]) / "revision_source.json"
        if not marker.exists() or read_json(marker).get("revision_source_digest") != expected_revision:
            continue
        path = Path(task["directory"]) / "test_summary.json"
        if not path.exists():
            continue
        summary = read_json(path)["summary"]
        if case["task"] == "latency":
            rows.append(dict(case, latency=summary))
            continue
        parent = manifest_parent(manifest, case["parent_id"])
        if parent is not None and "final_lambda" not in summary:
            final = Path(parent["directory"]) / "final_eval_summary.json"
            if final.exists():
                summary = dict(summary, final_lambda=read_json(final)["summary"].get("final_lambda"))
        rows.append(dict(case, **{k: summary.get(k) for k in METRICS}))
    return manifest, rows


def manifest_parent(manifest, identity):
    for parent in manifest["parents"]:
        if parent["id"] == identity:
            return parent
    return None


def fmt(value, digits=2):
    return "--" if value is None else f"{value:.{digits}f}"


def write_table(out, name, header, body, caption):
    with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(body)
    tex = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{tab:rev_{name}}}",
           r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", rf"\begin{{tabular}}{{l{'c' * (len(header) - 1)}}}",
           r"\hline", " & ".join(h.replace("%", r"\%").replace("_", r"\_") for h in header) + r" \\", r"\hline"]
    tex += [" & ".join(str(c).replace("%", r"\%").replace("_", r"\_") for c in row) + r" \\" for row in body]
    tex += [r"\hline", r"\end{tabular}", r"\end{table}"]
    (out / f"{name}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return md


def paired(rows, stage, variant, reference, metric, n=None):
    key = lambda r: (r["layout_seed"], r["seed"], r["N"])  # noqa: E731
    ref = {key(r): r[metric] for r in rows if r["experiment"] == stage and r["variant"] == reference
           and (n is None or r["N"] == n) and r[metric] is not None}
    deltas = [r[metric] - ref[key(r)] for r in rows if r["experiment"] == stage and r["variant"] == variant
              and key(r) in ref and r[metric] is not None and (n is None or r["N"] == n)]
    m, sd, k = mean_sd(deltas)
    if m is None:
        return "--"
    if k < 2:
        return f"{m:+.2f}"
    half = T95.get(k - 1, 1.96) * sd / math.sqrt(k)
    return f"{m:+.2f} [{m - half:+.2f}, {m + half:+.2f}]"


def interpolate_energy(points, target):
    """Energy at failure `target` by mixing the two neighbouring sweep points (episode-level time sharing)."""
    points = sorted(points)
    for (c1, e1), (c2, e2) in zip(points, points[1:]):
        if c1 <= target <= c2 and c2 > c1:
            return e1 + (e2 - e1) * (target - c1) / (c2 - c1)
    return None


def generate(root):
    root = Path(root)
    if not (root / "manifest.json").exists():
        return None
    manifest, rows = load(root)
    expected_revision = manifest.get("revision_source_digest", revision_source_digest())
    out = root / "report"
    out.mkdir(parents=True, exist_ok=True)
    md = ["# Revision study report", "",
          f"Source digest `{manifest['source_digest'][:16]}`; revision digest `"
          f"{manifest.get('revision_source_digest', expected_revision)[:16]}`. Means ± sample SD over paired deployments; "
          "paired deltas use nominal 95 % t intervals. A CI containing zero is reported as inconclusive evidence, not equivalence.", ""]
    for stage, columns in STAGE_COLUMNS.items():
        stage_rows = [r for r in rows if r["experiment"] == stage]
        if not stage_rows:
            continue
        groups = defaultdict(list)
        for r in stage_rows:
            groups[(r["variant"], r["N"])].append(r)
        reference = REFERENCE.get(stage)
        header = ["Variant", "N", "Pairs"] + [SHORT[c] for c in columns]
        if reference:
            header += [f"d Energy vs {reference}", f"d Failure vs {reference}"]
        body = []
        for (variant, n), items in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            row = [variant, n, len(items)]
            for c in columns:
                m, sd, k = mean_sd([r[c] for r in items])
                row.append("--" if m is None else (f"{m:.2f}±{sd:.2f}" if k > 1 else f"{m:.2f}"))
            if reference:
                row += ["--", "--"] if variant == reference else [
                    paired(rows, stage, variant, reference, "energy_mJ_device_slot", n),
                    paired(rows, stage, variant, reference, "cost_percent", n)]
            body.append(row)
        md += [f"## {stage}", ""] + write_table(out, stage, header, body, f"Revision study: {stage}.") + [""]
        if stage == "fair_critics":
            md += _matched_penalty(stage_rows) + [""]
    latency = [r for r in rows if r["experiment"] == "latency"]
    if latency:
        summary = latency[0]["latency"]
        header = ["Policy", "N", "Weights", "Actor parameters", "Median (ms)", "P95 (ms)", "P99 (ms)"]
        body = [[r["variant"], r["N"], r["weights"], r["parameters_actor"], fmt(r["median_ms"], 3),
                 fmt(r["p95_ms"], 3), fmt(r["p99_ms"], 3)] for r in summary["rows"]]
        md += [f"## latency ({summary['cpu']}, one thread)", ""] + write_table(
            out, "latency", header, body, f"Decision latency on {summary['cpu']}, one thread.") + [""]
    planned = len(manifest["cases"])
    md += [f"Completed cases: {len(rows)} of {planned}.", ""]
    (out / "REVISION_REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return out


def _matched_penalty(stage_rows):
    """Energy of the fixed-penalty sweep at SFR-SAC's failure level, per pair and averaged."""
    by_pair = defaultdict(lambda: dict(sweep=[], sfr=None))
    for r in stage_rows:
        key = (r["layout_seed"], r["seed"])
        if r["variant"].startswith("pen_struct_b") and r["cost_percent"] is not None:
            by_pair[key]["sweep"].append((r["cost_percent"], r["energy_mJ_device_slot"]))
        elif r["variant"] == "sfr":
            by_pair[key]["sfr"] = (r["cost_percent"], r["energy_mJ_device_slot"])
    gaps = []
    for key, value in by_pair.items():
        if value["sfr"] and len(value["sweep"]) >= 2:
            matched = interpolate_energy(value["sweep"], value["sfr"][0])
            if matched is not None:
                gaps.append(value["sfr"][1] - matched)
    m, sd, k = mean_sd(gaps)
    if m is None:
        return ["Matched-failure comparison with the structured fixed-penalty sweep: not available yet."]
    return [f"SFR-SAC energy minus structured fixed-penalty energy at SFR-SAC's failure level "
            f"(time-sharing between neighbouring beta values; betas {list(V.PENALTY_BETAS)}): "
            f"{m:+.2f} ± {sd:.2f} mJ over {k} pairs (negative = SFR-SAC better)."]
