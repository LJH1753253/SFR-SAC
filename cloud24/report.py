"""Reports derived only from completed runs, with deployment-aware summaries."""

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from v2.runtime import read_json, write_json

METRICS = ("cost_percent", "energy_mJ_device_slot", "bfsr_percent",
           "local_failure_percent", "edge_delivery_failure_percent", "mec_service_failure_percent",
           "E_loc_mJ_device_slot", "E_tx_mJ_device_slot", "avg_tx_power", "avg_offload_ratio",
           "final_lambda", "mean_lambda_last_window", "lambda_std_last_window",
           "best_ap_percent", "ap_load_cv", "task_failure_percent", "device_failure_p95_percent",
           "devices_above_budget", "policy_inference_ms_batch1", "parameters_actor")


def csv_write(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def raw_rows(root):
    rows = []
    for path in Path(root).glob("test/**/test_summary.json"):
        result = read_json(path)
        spec, summary = result["spec"], result["summary"]
        row = {key: spec[key] for key in ("variant", "method", "N", "layout_seed", "seed")}
        row.update(split="held_out_test", test_episodes=summary["eval_episodes"],
                   cost_budget=float(read_json(Path(spec["parent"])/"job.json")["config"]["cost_limit"]),
                   parent=spec["parent"])
        row.update({key: summary.get(key) for key in METRICS})
        rows.append(row)
    return sorted(rows, key=lambda r: (r["variant"], r["N"], r["layout_seed"], r["seed"]))


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["variant"], row["method"], row["N"], row["cost_budget"])].append(row)
    output = []
    for (variant, method, n, budget), runs in sorted(groups.items()):
        by_layout = defaultdict(list)
        for row in runs:
            by_layout[row["layout_seed"]].append(row)
        result = dict(variant=variant, method=method, N=n, cost_budget=budget,
                      completed_runs=len(runs), completed_deployments=len(by_layout),
                      completed_seeds=len({r["seed"] for r in runs}))
        for metric in METRICS:
            values = [float(np.mean([r[metric] for r in rs if r.get(metric) is not None]))
                      for rs in by_layout.values() if any(r.get(metric) is not None for r in rs)]
            result[metric] = float(np.mean(values)) if values else None
            result[metric+"_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
            result[metric+"_ci95_halfwidth"] = (critical.get(len(values), 1.96)*result[metric+"_std"]/math.sqrt(len(values))
                                                   if len(values) > 1 else None)
        output.append(result)
    return output


def matched_pairs(rows):
    pairs = []
    safe = [r for r in rows if r["method"] == "safe" and r["variant"] != "sac_lagrangian"]
    penalty = [r for r in rows if r["method"] == "penalty_sac"]
    for left in safe:
        for right in penalty:
            if any(left[k] != right[k] for k in ("N", "layout_seed", "seed")):
                continue
            gap = abs(left["cost_percent"]-right["cost_percent"])
            if gap <= .25:
                pairs.append(dict(N=left["N"], layout_seed=left["layout_seed"], seed=left["seed"],
                                  safe_variant=left["variant"], penalty_variant=right["variant"],
                                  cost_gap_pp=gap, safe_cost_percent=left["cost_percent"],
                                  penalty_cost_percent=right["cost_percent"],
                                  safe_energy=left["energy_mJ_device_slot"],
                                  penalty_energy=right["energy_mJ_device_slot"],
                                  energy_saving_percent=100*(1-left["energy_mJ_device_slot"]/right["energy_mJ_device_slot"])))
    return pairs


def generate(root, final=False):
    root = Path(root)
    rows = raw_rows(root)
    combined = aggregate(rows)
    csv_write(root/"raw_test_results.csv", rows, list(rows[0]) if rows else ["variant", "N", "seed", "completed_seeds"])
    csv_write(root/"summary_latest.csv", combined, list(combined[0]) if combined else ["variant", "N", "completed_seeds"])
    csv_write(root/"matched_failure_pairs.csv", matched_pairs(rows))
    validation = []
    for path in root.glob("runs/**/final_eval_summary.json"):
        result = read_json(path)
        task_path = path.parent/"task.json"
        if not task_path.exists():
            continue
        task = read_json(task_path)
        validation.append(dict(variant=task["variant"], N=task["N"], layout_seed=task["layout_seed"],
                               seed=task["seed"], split="validation", **{k: result["summary"].get(k) for k in METRICS}))
    csv_write(root/"raw_validation_results.csv", validation)
    state = read_json(root/"suite_status.json") if (root/"suite_status.json").exists() else {}
    selection = read_json(root/"selection_locked.json") if (root/"selection_locked.json").exists() else None
    lines = ["# Cloud study progress", "", f"Phase: {state.get('phase', 'initializing')}",
             f"Status: {state.get('status', 'initializing')}",
             f"Completed validation/training artifacts: {len(validation)}",
             f"Completed held-out tests: {len(rows)}", "",
             "Environment: physical CPU-cycle energy and PA/circuit TDMA-airtime energy; eta=0.30; E_ref=0.02 J.",
             "Default budget d=0.10; trade-off arms explicitly change d or beta.",
             "Energy: mJ/device/slot. Cost and BFSR: percent.",
             "Five paired deployments are planned for the main comparison; completed counts below are actual.",
             "Statistics average within deployments first. Slots are not treated as independent training replicates.", ""]
    if selection:
        lines.extend([f"Locked selection: {selection['selected']}; promoted={selection['promoted']}", selection["reason"], ""])
    else:
        lines.extend(["No candidate is locked yet. No held-out test is used for selection.", ""])
    if validation:
        validation_groups = defaultdict(list)
        for row in validation:
            if row["variant"] in {"shared_exact", "split_exact", "split_bounded", "split_values"} and row["N"] == 70:
                validation_groups[row["variant"]].append(row)
        lines.extend(["## Completed N70 validation runs (not final test)", "",
                      "| Variant | Runs | Cost (%) | Energy (mJ/device/slot) | BFSR (%) |",
                      "|---|---:|---:|---:|---:|"])
        for variant, values in sorted(validation_groups.items()):
            means = {metric: np.mean([r[metric] for r in values]) for metric in
                     ("cost_percent", "energy_mJ_device_slot", "bfsr_percent")}
            lines.append(f"| {variant} | {len(values)} | {means['cost_percent']:.3f} | "
                         f"{means['energy_mJ_device_slot']:.3f} | {means['bfsr_percent']:.3f} |")
        lines.extend(["", "Validation means are descriptive; confirmation separately requires a complete 3x3 grid.", ""])
    lines.extend(["## Held-out test results", ""])
    lines.extend(["| Variant | N | Deployments | Runs | Cost (%) | Energy (mJ/device/slot) | BFSR (%) |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for row in combined:
        lines.append(f"| {row['variant']} | {row['N']} | {row['completed_deployments']} | {row['completed_runs']} | "
                     f"{row['cost_percent']:.3f} | {row['energy_mJ_device_slot']:.3f} | {row['bfsr_percent']:.3f} |")
    lines.extend(["", "The DPP baseline is a Lyapunov-inspired rule, not an exact drift-plus-penalty optimizer.",
                  "MB-greedy uses full simulator-model information, including device energy coefficients; neural actors do not.",
                  "The critics are immediate predictors (gamma=0); this study does not establish a long-horizon CMDP safety guarantee.",
                  "An energy/failure trade-off or a low mean cost alone is not proof of Pareto superiority or per-device reliability.",
                  "All failed and interrupted task states remain on disk."])
    (root/"partial_report_latest.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    table = [r"\begin{tabular}{lrrrr}", r"\hline", r"Method & $N$ & Cost (\%) & Energy (mJ/device/slot) & BFSR (\%) \\"]
    for row in combined:
        name = row["variant"].replace("_", r"\_")
        cells = [f"${row[k]:.2f}" + (f" \\pm {row[k+'_std']:.2f}" if row[k+"_std"] is not None else "") + "$"
                 for k in ("cost_percent", "energy_mJ_device_slot", "bfsr_percent")]
        table.append(f"{name} & {row['N']} & " + " & ".join(cells) + " " + r"\\")
    table.extend([r"\hline", r"\end{tabular}"])
    (root/"summary_latest.tex").write_text("\n".join(table)+"\n", encoding="utf-8")
    if final:
        plots(root, combined)


def plots(root, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "pdf.fonttype": 42})
    figure_dir = root/"figures"
    figure_dir.mkdir(exist_ok=True)
    if not summary:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for variant in sorted({r["variant"] for r in summary}):
        group = sorted([r for r in summary if r["variant"] == variant], key=lambda r: r["N"])
        if len(group) < 2:
            continue
        for axis, metric, label in zip(axes, ("cost_percent", "energy_mJ_device_slot", "bfsr_percent"),
                                       ("Service-failure cost (%)", "Avg. device energy (mJ/device/slot)", "BFSR (%)")):
            x, y = [r["N"] for r in group], [r[metric] for r in group]
            error = [r[metric+"_ci95_halfwidth"] or 0 for r in group]
            axis.plot(x, y, marker="o", label=variant.replace("_", " "))
            lo, hi = np.asarray(y)-error, np.asarray(y)+error
            if metric.endswith("percent"):
                lo, hi = np.maximum(lo, 0), np.minimum(hi, 100)
            axis.fill_between(x, lo, hi, alpha=.10)
            axis.set(xlabel="Number of devices", ylabel=label)
    axes[0].axhline(10, color="black", linestyle="--", linewidth=.8)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncols=4, fontsize=8)
    fig.savefig(figure_dir/"main_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    group = [r for r in summary if r["N"] == 70]
    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for row in group:
        axis.scatter(row["cost_percent"], row["energy_mJ_device_slot"], label=row["variant"].replace("_", " "))
    axis.set(xlabel="Service-failure cost (%)", ylabel="Energy (mJ/device/slot)")
    axis.legend(fontsize=7, ncols=2)
    fig.savefig(figure_dir/"energy_failure_N70.pdf")
    plt.close(fig)
    selected = read_json(root/"selection_locked.json")["selected"] if (root/"selection_locked.json").exists() else "shared_exact"
    displayed = (selected, "sac_lagrangian", "penalty_beta5", "td3", "dpp_rule", "vanilla")
    for filename, variants, metrics in (
        ("constrained_comparison.pdf", displayed[:-1], ("cost_percent", "energy_mJ_device_slot", "bfsr_percent")),
        ("safe_vs_vanilla.pdf", (selected, "vanilla"), ("cost_percent", "energy_mJ_device_slot")),
    ):
        fig, axes = plt.subplots(1, len(metrics), figsize=(4.3*len(metrics), 3.6), constrained_layout=True)
        for variant in variants:
            points = sorted([r for r in summary if r["variant"] == variant], key=lambda r: r["N"])
            if not points:
                continue
            for axis, metric in zip(axes, metrics):
                axis.errorbar([r["N"] for r in points], [r[metric] for r in points],
                              yerr=[r[metric+"_std"] or 0 for r in points], marker="o", capsize=2,
                              label="Lyapunov-inspired" if variant == "dpp_rule" else variant.replace("_", " "))
                axis.set(xlabel="Number of devices", ylabel={"cost_percent": "Service-failure cost (%)",
                         "energy_mJ_device_slot": "Energy (mJ/device/slot)", "bfsr_percent": "BFSR (%)"}[metric])
                if metric == "bfsr_percent":
                    axis.set_ylim(0, 100)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="outside lower center", ncols=3, fontsize=8)
        fig.savefig(figure_dir/filename, bbox_inches="tight")
        plt.close(fig)
    components = [next((r for r in group if r["variant"] == v), None) for v in displayed]
    components = [r for r in components if r is not None]
    if components:
        fig, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
        bottom = np.zeros(len(components))
        for metric, label in (("local_failure_percent", "Local-computing failure"),
                              ("edge_delivery_failure_percent", "Edge-delivery failure"),
                              ("mec_service_failure_percent", "MEC-service failure")):
            values = np.asarray([r[metric] for r in components])
            axis.bar(np.arange(len(components)), values, bottom=bottom, label=label)
            bottom += values
        axis.set(xticks=np.arange(len(components)), xticklabels=[r["variant"].replace("_", " ") for r in components],
                 ylabel="Service-failure components (%)", title="N=70")
        axis.tick_params(axis="x", rotation=20)
        axis.legend(fontsize=8)
        fig.savefig(figure_dir/"failure_components_N70.pdf")
        plt.close(fig)
    for metric, label in (("lambda", r"$\lambda$"), ("train_ap_load_cv", "AP load coefficient of variation")):
        fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
        paths = list(root.glob("runs/N70/layout1000/train0/*/training.csv"))
        for path in paths:
            with path.open(encoding="utf-8") as stream:
                records = list(csv.DictReader(stream))
            points = [(int(r["episode"]), float(r[metric])) for r in records if r.get(metric) not in {None, ""}]
            if not points:
                continue
            variant = read_json(path.parent/"task.json")["variant"]
            if variant not in {"shared_exact", "split_exact", "split_bounded", "split_values",
                                "split_fixed_prior", "no_pooled_context", "mlp_critics"}:
                continue
            x, y = zip(*points)
            axis.plot(x, y, label=variant.replace("_", " "), linewidth=1)
        axis.set(xlabel="Training episode", ylabel=label, title="N=70, layout 1000, training seed 0")
        handles, _ = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=7, ncols=2)
        fig.savefig(figure_dir/("lambda_curves.pdf" if metric == "lambda" else "ap_load_curves.pdf"))
        plt.close(fig)
