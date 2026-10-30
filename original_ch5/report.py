import csv
import math
from pathlib import Path

import numpy as np

from v2.runtime import read_json, write_json
from original_ch5.protocol import BLOCKED_COST, EXPECTED_SOURCE, addon_digest


METRICS = ("cost_percent", "energy_mJ_device_slot", "bfsr_percent", "local_failure_percent",
           "edge_delivery_failure_percent", "mec_service_failure_percent", "avg_tx_power",
           "final_lambda", "mean_lambda_last_window", "lambda_std_last_window",
           "cost_p95_percent", "device_failure_p95_percent", "device_failure_max_percent",
           "devices_above_budget", "task_failure_percent", "jain_completion")
LABELS = {"split_exact": "Safe SAC", "safe": "Safe SAC", "penalty_beta5": "Penalty SAC",
          "td3": "Safe TD3", "dpp_rule": "Lyapunov-inspired rule", "vanilla": "Vanilla SAC"}


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def generate(root, final=False):
    root = Path(root)
    manifest = root / "manifest.json"
    if not manifest.exists():
        return
    plan = read_json(manifest)
    tests = {s["id"]: s for s in plan["evaluations"]}
    parents = {s["id"]: s for s in plan["parents"]}
    rows, groups = [], {}
    for case in plan["cases"]:
        key = (case["experiment"], case["variant"], case["agent"], case["N"])
        groups.setdefault(key, [])
        spec = tests[case["test_id"]]
        path = Path(spec["directory"]) / "test_summary.json"
        if not path.exists():
            continue
        result = read_json(path)
        if result["id"] != spec["id"] or result["source_digest"] != EXPECTED_SOURCE or result["addon_digest"] != addon_digest():
            raise RuntimeError(f"Mismatched test result: {path}")
        row = {**case, **result["summary"], "completed_seeds": 1,
               "source": str(path), "reused_model": parents[case["parent_id"]]["reused"]}
        rows.append(row)
        groups[key].append(row)
    summary = []
    for key, values in sorted(groups.items()):
        experiment, variant, agent, n = key
        sample = values[0] if values else next(c for c in plan["cases"] if
                  (c["experiment"], c["variant"], c["agent"], c["N"]) == key)
        item = dict(experiment=experiment, variant=variant, agent=agent, N=n,
                    eta=sample["eta"], d=sample["d"], b0=sample["b0"], beta=sample["beta"],
                    completed_seeds=len(values), expected_seeds=5 if experiment == "main" else 3)
        for metric in METRICS:
            numbers = [float(v[metric]) for v in values if v.get(metric) is not None]
            item[metric+"_mean"] = float(np.mean(numbers)) if numbers else None
            item[metric+"_std"] = float(np.std(numbers, ddof=1)) if len(numbers) > 1 else None
        summary.append(item)
    csv_write(root / "raw.csv", rows)
    csv_write(root / "summary.csv", summary)
    csv_write(root / "partial_summary_latest.csv", summary)
    for experiment in {r["experiment"] for r in summary}:
        csv_write(root / experiment / "raw.csv", [r for r in rows if r["experiment"] == experiment])
        subset = [r for r in summary if r["experiment"] == experiment]
        csv_write(root / experiment / "summary.csv", subset)
        tex = ["\\begin{tabular}{llrrr}", "\\hline",
               "Variant & Completed & Cost (\\%) & Energy (mJ/device/slot) & BFSR (\\%) \\\\", "\\hline"]
        for r in subset:
            def fmt(metric):
                mean, std = r.get(metric+"_mean"), r.get(metric+"_std")
                if mean is None:
                    return "--"
                return f"{mean:.2f}" + (f" $\\pm$ {std:.2f}" if std is not None else "")
            label = LABELS.get(r["variant"], r["variant"]).replace("_", "\\_") + f" ($N={r['N']}$)"
            tex.append(f"{label} & {r['completed_seeds']}/{r['expected_seeds']} & {fmt('cost_percent')} & "
                       f"{fmt('energy_mJ_device_slot')} & {fmt('bfsr_percent')} \\\\")
        tex.extend(["\\hline", "\\end{tabular}", ""])
        (root / experiment / "summary.tex").write_text("\n".join(tex), encoding="utf-8")
    state = read_json(root / "suite_status.json") if (root / "suite_status.json").exists() else {}
    lines = ["# Original Chapter V completion", "", f"Phase: {state.get('phase', 'planned')}",
             f"Status: {state.get('status', 'planned')}",
             "", "Fixed policy: split_exact. No policy/default reselection.",
             "Energy: mJ/device/slot. Cost and BFSR: percent.",
             "Main: five paired deployment/training replicates. Ablations: three pairs. Slots are not training replicates.",
             "Intervals shown in tables are sample standard deviations across the completed pairs.",
             "", "| Experiment | Variant | N | Completed seeds | Cost (%) | Energy | BFSR (%) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        def number(key):
            value = r.get(key+"_mean")
            return "--" if value is None else f"{value:.3f}"
        lines.append(f"| {r['experiment']} | {r['variant']} | {r['N']} | "
                     f"{r['completed_seeds']}/{r['expected_seeds']} | {number('cost_percent')} | "
                     f"{number('energy_mJ_device_slot')} | {number('bfsr_percent')} |")
    lines += ["", "## Not scheduled", "", BLOCKED_COST,
              "", "No new beta grid, action repair, automatic winner selection, or extra baseline diagnosis is run.",
              "Revision extensions (CPO/reward horizon/hinge/AP/latency) are not implemented by this increment.",
              "Dual variants use equal nominal learning rates, not equal total dual gain per episode.",
              "The Lyapunov reference is a heuristic, not an exact DPP optimizer.",
              "Aggregate budget satisfaction does not imply per-device reliability or hard slot safety.",
              "Final evaluation uses one new fixed stream for every method; old test-stream means are not spliced into it.",
              "Compact exports omit full replay checkpoints. Keep the remote output directory to resume interrupted training."]
    (root / "partial_report_latest.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    write_json(root / "completion.json", dict(completed_case_rows=len(rows), planned_case_rows=len(plan["cases"]),
               unique_evaluations_done=len({r["test_id"] for r in rows}), unique_evaluations_planned=len(tests),
               blocked_experiments=plan["protocol"]["blocked_experiments"]))
    if final:
        (root / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        plot(root, summary, rows, parents)


def plot(root, summary, rows, parents):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.bbox": "tight"})
    ready = [r for r in summary if r["completed_seeds"] == r["expected_seeds"]]
    def series_figure(items, variants, metrics, name, xkey="N"):
        if not items:
            return
        fig, axes = plt.subplots(1, len(metrics), figsize=(4*len(metrics), 3.1), squeeze=False,
                                 constrained_layout=True)
        for ax, (metric, title) in zip(axes[0], metrics):
            for variant in variants:
                values = sorted([r for r in items if r["variant"] == variant], key=lambda r: r[xkey])
                if not values:
                    continue
                x = [r[xkey] for r in values]
                y = [r[metric+"_mean"] for r in values]
                error = [r[metric+"_std"] or 0 for r in values]
                ax.errorbar(x, y, yerr=error, marker="o", capsize=3,
                            label=LABELS.get(variant, variant))
            ax.set_xlabel("Number of EDs, N" if xkey == "N" else xkey)
            ax.set_ylabel(title)
            ax.grid(alpha=.25)
            if metric == "bfsr_percent":
                ax.set_ylim(0, 100)
            if xkey == "eta":
                ax.set_xticks([i/10 for i in range(1, 10)])
                ax.axvline(.3, ls=":", color="black", label="Default eta=0.30")
        axes[0, 0].legend(fontsize=8)
        fig.savefig(root / name)
        plt.close(fig)
    main = [r for r in ready if r["experiment"] == "main"]
    series_figure(main, ["split_exact", "penalty_beta5", "td3", "dpp_rule"],
                  [("cost_percent", "Service-failure cost (%)"),
                   ("energy_mJ_device_slot", "Energy (mJ/device/slot)"),
                   ("bfsr_percent", "BFSR (%)")], "Fig_main_constrained_comparison.pdf")
    series_figure(main, ["split_exact", "vanilla"],
                  [("cost_percent", "Service-failure cost (%)"),
                   ("energy_mJ_device_slot", "Energy (mJ/device/slot)")], "Fig_safe_vs_vanilla.pdf")
    components = [r for r in main if r["N"] == 70]
    if len(components) == 5:
        fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
        order = ["split_exact", "penalty_beta5", "td3", "dpp_rule", "vanilla"]
        data = {r["variant"]: r for r in components}
        bottom = np.zeros(5)
        for key, label in [("local_failure_percent", "Local-computing failure"),
                           ("edge_delivery_failure_percent", "Edge-delivery failure"),
                           ("mec_service_failure_percent", "MEC-service failure")]:
            values = np.array([data[v][key+"_mean"] for v in order])
            ax.bar(range(5), values, bottom=bottom, label=label)
            bottom += values
        ax.set_xticks(range(5), [LABELS[v] for v in order], rotation=15, ha="right")
        ax.set_ylabel("Service-failure cost (%)")
        ax.legend(fontsize=8)
        fig.savefig(root / "Fig_failure_components_N70.pdf")
        plt.close(fig)
    for stage, xkey in [("budget", "d"), ("eta", "eta"), ("penalty", "beta"), ("background", "b0")]:
        items = [r for r in ready if r["experiment"] == stage]
        if items:
            # One curve connects the declared parameter values, not separate one-point series.
            items = [{**r, "variant": stage} for r in items]
            series_figure(items, [stage], [("cost_percent", "Service-failure cost (%)"),
                          ("energy_mJ_device_slot", "Energy (mJ/device/slot)"), ("bfsr_percent", "BFSR (%)")],
                          f"Fig_{stage}_sensitivity.pdf", xkey=xkey)
    dual = [r for r in rows if r["experiment"] == "dual"]
    if dual:
        fig, ax = plt.subplots(figsize=(5.5, 3.3), constrained_layout=True)
        for mode in ("batch", "episode", "step"):
            histories = []
            for row in dual:
                if row["variant"] != mode:
                    continue
                path = Path(parents[row["parent_id"]]["directory"]) / "training.csv"
                with path.open(encoding="utf-8-sig") as stream:
                    histories.append([float(r["lambda"]) for r in csv.DictReader(stream)])
            if len(histories) == 3 and len({len(h) for h in histories}) == 1:
                arr = np.asarray(histories)
                mean, std = arr.mean(0), arr.std(0, ddof=1)
                x = np.arange(1, len(mean)+1)
                ax.plot(x, mean, label=mode)
                ax.fill_between(x, np.maximum(mean-std, .01), mean+std, alpha=.12)
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Lagrange multiplier")
        if ax.lines:
            ax.legend()
            fig.savefig(root / "Fig_lambda_update_curves.pdf")
        plt.close(fig)
