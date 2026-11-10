"""Generate the final Paper1 figures from results/paper only."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "paper"
TABLES = RESULTS / "tables"
PER_SEED = RESULTS / "per_seed"
SUMMARY = RESULTS / "summary"
FIGURES = RESULTS / "figures"

plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def rows(name: str, folder: Path = TABLES):
    with (folder / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value: str) -> float:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    if not match:
        raise ValueError(f"No number in {value!r}")
    return float(match.group(0))


def mean_sd(value: str):
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", str(value))
    if len(matches) >= 2:
        return float(matches[0]), float(matches[1])
    return float(matches[0]), 0.0


def save(fig, name: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    if name == "Fig1_main_performance":
        fig.subplots_adjust(left=.06, right=.99, bottom=.18, top=.90, wspace=.30)
    else:
        fig.tight_layout(pad=1.0)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main_results():
    data = rows("main_results.csv")
    algorithms = ["SFR-SAC", "Penalty SAC", "Safe TD3", "Lyapunov-inspired rule", "Vanilla SAC"]
    labels = {"SFR-SAC": "SFR-SAC", "Penalty SAC": "Penalty SAC", "Safe TD3": "Safe TD3",
              "Lyapunov-inspired rule": "Lyapunov-inspired rule", "Vanilla SAC": "Vanilla SAC"}
    colors = {"SFR-SAC": "#1f77b4", "Penalty SAC": "#2ca02c", "Safe TD3": "#7f7f7f",
              "Lyapunov-inspired rule": "#9467bd", "Vanilla SAC": "#d62728"}
    metrics = [("Cost (\\%)", "Service-failure cost (\\%)"),
               ("Energy (mJ/device/slot)", "Avg. device energy (mJ/device/slot)"),
               ("BFSR (\\%)", "BFSR (\\%)")]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.15))
    ns = sorted({int(r["N"]) for r in data})
    for ax, (field, title) in zip(axes, metrics):
        for algorithm in algorithms:
            subset = sorted((r for r in data if r["Algorithm"] == algorithm), key=lambda r: int(r["N"]))
            x = np.array([int(r["N"]) for r in subset])
            y, e = zip(*(mean_sd(r[field]) for r in subset))
            ax.errorbar(x, y, yerr=e, marker="o", linewidth=1.5, markersize=3.5,
                        capsize=2, label=labels[algorithm], color=colors[algorithm])
        ax.set_title(title)
        ax.set_xticks(ns)
        ax.grid(alpha=.25)
    # Keep the full Vanilla SAC range in the main panel and add a clearly
    # labelled zoom for the constrained operating range.
    zoom = inset_axes(axes[0], width="48%", height="43%", loc="upper left", borderpad=1.0)
    for algorithm in algorithms[:-1]:
        subset = sorted((r for r in data if r["Algorithm"] == algorithm), key=lambda r: int(r["N"]))
        x = np.array([int(r["N"]) for r in subset])
        y, e = zip(*(mean_sd(r["Cost (\\%)"]) for r in subset))
        zoom.errorbar(x, y, yerr=e, marker="o", linewidth=1.0, markersize=2.5,
                      capsize=1.5, color=colors[algorithm])
    zoom.axhline(10, color="black", linestyle="--", linewidth=.6)
    zoom.set_ylim(0, 12)
    zoom.set_title("Zoom: cost <= 12%", fontsize=7)
    zoom.set_xticks([50, 60, 70])
    zoom.tick_params(labelsize=6)
    axes[0].axhline(10, color="black", linestyle="--", linewidth=.8, label="Budget d=10%")
    axes[0].set_ylabel("Percent")
    axes[1].set_ylabel("mJ/device/slot")
    axes[2].set_ylabel("Percent")
    axes[2].legend(frameon=False, loc="best")
    save(fig, "Fig1_main_performance")


def fair_critics():
    data = rows("fair_critics.csv")
    wanted = ["sfr", "td3", "td3_struct", "td3_struct_split"]
    labels = ["SFR-SAC", "Safe TD3", "TD3+SC", "TD3+SC+SA"]
    colors = ["#1f77b4", "#7f7f7f", "#d62728", "#ff7f0e"]
    selected = [next(r for r in data if r["variant"] == key) for key in wanted]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.15))
    metrics = [("cost_percent_mean", "cost_percent_sd", "Service-failure cost (\\%)"),
               ("energy_mJ_device_slot_mean", "energy_mJ_device_slot_sd", "Avg. device energy (mJ/device/slot)"),
               ("bfsr_percent_mean", "bfsr_percent_sd", "BFSR (\\%)")]
    x = np.arange(len(selected))
    for ax, (mean_key, sd_key, title) in zip(axes, metrics):
        ax.bar(x, [float(r[mean_key]) for r in selected], yerr=[float(r[sd_key]) for r in selected],
               capsize=3, color=colors, edgecolor="black", linewidth=.4)
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.grid(axis="y", alpha=.25)
    axes[0].axhline(10, color="black", linestyle="--", linewidth=.8)
    axes[0].set_ylabel("Percent")
    axes[1].set_ylabel("mJ/device/slot")
    axes[2].set_ylabel("Percent")
    save(fig, "Fig2_fair_critics")


def penalty_form():
    data = rows("penalty_form.csv")
    order = ["sfr_hinge_d0.08", "sfr_hinge_d0.1", "sfr_hinge_d0.12",
             "sfr_linear_d0.08", "sfr_linear_d0.1", "sfr_linear_d0.12"]
    selected = [next(r for r in data if r["Variant"] == key) for key in order]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.1))
    budgets = [8, 10, 12]
    for ax, field, title in zip(axes, ["Failure (%)", "Energy (mJ)", "BFSR (%)"],
                                ["Cost", "Energy", "BFSR"]):
        for i, mode in enumerate(("hinge", "linear")):
            group = [r for r in selected if mode in r["Variant"]]
            means, errors = zip(*(mean_sd(r[field]) for r in group))
            ax.errorbar(x + (i - .5) * .08, means, yerr=errors, marker="o", capsize=2,
                        linewidth=1.4, label=mode.capitalize())
        ax.set_title(title)
        ax.set_xticks(x, ["d=0.08", "d=0.10", "d=0.12"])
        ax.grid(alpha=.25)
    for i, budget in enumerate(budgets):
        axes[0].hlines(budget, i - .16, i + .16, color="black", linestyle="--", linewidth=.8)
    axes[0].plot([], [], color="black", linestyle="--", label="Budget target d")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("Percent")
    axes[1].set_ylabel("mJ/device/slot")
    axes[2].legend(frameon=False)
    save(fig, "Fig3_penalty_form")


def per_slot():
    data = rows("per_slot.csv")
    labels = ["Actor", "Critic refine", "Critic scratch", "Numerical optimizer"]
    selected = [next(r for r in data if r["method"] == label) for label in labels]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for r, color in zip(selected, colors):
        energy = float(r["energy_mJ_device_slot_mean"])
        cost = float(r["cost_percent_mean"])
        axes[0].errorbar(energy, cost,
                         xerr=float(r["energy_mJ_device_slot_sd"]), yerr=float(r["cost_percent_sd"]),
                         marker="o", markersize=5, capsize=2, label=r["method"], color=color)
        axes[0].annotate(f"BFSR {float(r['bfsr_percent_mean']):.1f}%", (energy, cost),
                         xytext=(4, 5), textcoords="offset points", fontsize=6, color=color)
    axes[0].axhline(10, color="black", linestyle="--", linewidth=.8, label="Budget d=10%")
    axes[0].set_xlabel("Avg. device energy (mJ/device/slot)")
    axes[0].set_ylabel("Service-failure cost (\\%)")
    axes[0].set_title("Energy--failure operating points")
    axes[0].grid(alpha=.25)
    x = np.arange(len(selected))
    axes[1].bar(x, [float(r["decision_ms_median_mean"]) for r in selected],
                yerr=[float(r["decision_ms_median_sd"]) for r in selected], capsize=2, color=colors)
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, ["Actor", "Refine", "Scratch", "Numerical\noptimizer"], rotation=15, ha="right")
    axes[1].set_ylabel("Median decision time (ms, log scale)")
    axes[1].set_title("Online decision latency")
    axes[1].grid(axis="y", alpha=.25)
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, "Fig4_per_slot")


def ap_ablation():
    data = rows("ap.csv")
    wanted = ["sfr", "sfr_maxgain", "sfr_apind", "td3_apind", "mb_greedy_apind"]
    labels = ["SFR Learned", "SFR Best-AP", "SFR AP-ind.",
              "TD3 AP-ind.", "MB AP-ind."]
    selected = [next(r for r in data if r["Variant"] == key) for key in wanted]
    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#7f7f7f", "#2ca02c"]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.25))
    x = np.arange(len(selected))
    for ax, field, title, ylabel in zip(axes, ["Failure (%)", "Energy (mJ)", "AP-load CV"],
                                        ["Service-failure cost", "Energy", "AP-load concentration"],
                                        ["Percent", "mJ/device/slot", "Coefficient of variation"]):
        means, errors = zip(*(mean_sd(r[field]) for r in selected))
        ax.bar(x, means, yerr=errors, capsize=2, color=colors, edgecolor="black", linewidth=.35)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.grid(axis="y", alpha=.25)
    axes[0].axhline(10, color="black", linestyle="--", linewidth=.8)
    save(fig, "Fig5_ap_association")


def mismatch():
    data = rows("mismatch.csv")
    cases = [("mec_capacity", "MEC capacity"), ("pa_efficiency", "PA efficiency"),
             ("cycles", "CPU cycles/bit"), ("mec_sharing", "MEC sharing")]
    arms = ["sfr_wrong", "sfr_right", "td3", "mb"]
    labels = {"sfr_wrong": "SFR wrong", "sfr_right": "SFR correct", "td3": "Safe TD3", "mb": "MB-greedy"}
    fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.2), sharey="row")
    for col, (case, title) in enumerate(cases):
        selected = []
        names = []
        for arm in arms:
            candidates = [r for r in data if r["Variant"] == f"mm_{case}_{arm}"]
            if not candidates:
                continue
            selected.append(candidates[0])
            names.append(labels[arm])
        x = np.arange(len(selected))
        for row, field, ylabel in ((0, "Failure (%)", "Cost (\\%)"), (1, "BFSR (%)", "BFSR (\\%)")):
            vals = [mean_sd(r[field]) for r in selected]
            axes[row, col].bar(x, [v[0] for v in vals], yerr=[v[1] for v in vals], capsize=2,
                               color=["#d62728" if "wrong" in n.lower() else "#1f77b4" if "SFR" in n else "#7f7f7f" for n in names])
            axes[row, col].set_xticks(x, names, rotation=30, ha="right", fontsize=6.5)
            axes[row, col].grid(axis="y", alpha=.25)
            if col == 0:
                axes[row, col].set_ylabel(ylabel)
            if row == 0:
                axes[row, col].set_title(title)
        axes[0, col].axhline(10, color="black", linestyle="--", linewidth=.7)
    save(fig, "Fig6_model_mismatch")


def cpo():
    data = rows("cpo.csv")
    main = rows("main_results.csv")
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.1))
    main_rows = sorted((r for r in main if r["Algorithm"] == "SFR-SAC"), key=lambda r: int(r["N"]))
    x_main = [int(r["N"]) for r in main_rows]
    for ax, field, title in zip(axes, ["Cost (\\%)", "Energy (mJ/device/slot)", "BFSR (\\%)"],
                                ["Cost", "Energy", "BFSR"]):
        y, e = zip(*(mean_sd(r[field]) for r in main_rows))
        ax.errorbar(x_main, y, yerr=e, marker="D", color="#1f77b4", linewidth=1.6,
                    capsize=2, label="SFR-SAC")
        ax.set_title(title)
        ax.grid(alpha=.25)
    for variant, marker, color in (("cpo_g0", "o", "#1f77b4"), ("cpo_g0.99", "s", "#d62728")):
        selected = sorted((r for r in data if r["Variant"] == variant), key=lambda r: int(r["N"]))
        x = [int(r["N"]) for r in selected]
        for ax, field, title in zip(axes, ["Failure (%)", "Energy (mJ)", "BFSR (%)"], ["Cost", "Energy", "BFSR"]):
            y, e = zip(*(mean_sd(r[field]) for r in selected))
            ax.errorbar(x, y, yerr=e, marker=marker, color=color, linewidth=1.4, capsize=2,
                        label=r"$\gamma=" + ("0" if variant == "cpo_g0" else "0.99") + r"$")
            ax.grid(alpha=.25)
    axes[0].axhline(10, color="black", linestyle="--", linewidth=.8)
    axes[0].set_ylabel("Percent")
    axes[1].set_ylabel("mJ/device/slot")
    axes[2].legend(frameon=False, title="Split-actor CPO / SFR-SAC")
    save(fig, "Fig7_cpo")


def horizon():
    data = rows("horizon.csv")
    families = [("sacmlp", "SAC MLP"), ("td3", "Safe TD3"), ("pen5", "Penalty SAC")]
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.1))
    x = np.arange(len(families))
    width = .34
    for gamma, color, offset in (("g0", "#1f77b4", -width / 2), ("g0.99", "#d62728", width / 2)):
        selected = []
        for prefix, _ in families:
            selected.append(next(r for r in data if r["Variant"] == f"{prefix}_{gamma}"))
        for ax, field, title in zip(axes, ["Failure (%)", "Energy (mJ)", "BFSR (%)"], ["Cost", "Energy", "BFSR"]):
            y, e = zip(*(mean_sd(r[field]) for r in selected))
            ax.bar(x + offset, y, width=width, yerr=e, capsize=2, color=color,
                   edgecolor="black", linewidth=.35,
                   label=rf"$\gamma={0 if gamma == 'g0' else 0.99}$")
            ax.grid(alpha=.25)
    axes[0].set_xticks(x, [label for _, label in families], rotation=15, ha="right")
    axes[1].set_xticks(x, [label for _, label in families], rotation=15, ha="right")
    axes[2].set_xticks(x, [label for _, label in families], rotation=15, ha="right")
    axes[0].axhline(10, color="black", linestyle="--", linewidth=.8)
    axes[0].set_ylabel("Percent")
    axes[1].set_ylabel("mJ/device/slot")
    axes[2].legend(frameon=False)
    save(fig, "Fig8_horizon")


def latency():
    data = rows("latency.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    styles = (("Median (ms)", "-", "o"), ("P95 (ms)", "--", "s"), ("P99 (ms)", ":", "^"))
    for policy, color, marker in (("sfr", "#1f77b4", "o"), ("td3", "#7f7f7f", "s")):
        selected = [r for r in data if r["Policy"] == policy and r["Weights"] == "initialized"]
        selected.sort(key=lambda r: int(r["N"]))
        x = [int(r["N"]) for r in selected]
        for column, linestyle, point in styles:
            ax.plot(x, [float(r[column]) for r in selected], marker=point, linestyle=linestyle,
                    color=color, label=f"{policy.upper()} initialized {column.split()[0]}")
    trained = next(r for r in data if r["Policy"] == "sfr" and r["Weights"] == "trained")
    for column, linestyle, point in styles:
        ax.scatter([70], [float(trained[column])], marker=point, s=42, color="#d62728",
                   label=f"SFR-SAC trained P{column.split()[0]}")
    ax.set_yscale("log")
    ax.set_xlabel("Number of devices N")
    ax.set_ylabel("Policy/action-decode time (ms, log scale)")
    ax.set_title("Decision latency: initialized vs trained policy")
    ax.grid(alpha=.25)
    ax.legend(frameon=False, ncol=3, fontsize=6.5)
    save(fig, "Fig9_latency_scaling")


def sensitivity():
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0))
    for ax, filename, prefix, title in (
        (axes[0, 0], "eta_sensitivity_N70.csv", "eta", "Service-window fraction eta"),
        (axes[0, 1], "budget_sensitivity_N70.csv", "d", "Budget d"),
        (axes[1, 0], "background_load_N70.csv", "b0", "MEC background load"),
        (axes[1, 1], "penalty_beta_N70.csv", "beta", "Fixed penalty beta"),
    ):
        data = rows(filename)
        x = [number(r["Variant"].replace(prefix, "", 1)) for r in data]
        data = [row for _, row in sorted(zip(x, data), key=lambda pair: pair[0])]
        x = [number(r["Variant"].replace(prefix, "", 1)) for r in data]
        cost_field = next(key for key in data[0] if key.startswith("Cost"))
        cost = [mean_sd(r[cost_field]) for r in data]
        energy_field = next(key for key in data[0] if key.startswith("Energy"))
        energy = [mean_sd(r[energy_field]) for r in data]
        ax.errorbar(x, [v[0] for v in cost], yerr=[v[1] for v in cost], marker="o", label="Cost (%)")
        ax2 = ax.twinx()
        ax2.errorbar(x, [v[0] for v in energy], yerr=[v[1] for v in energy], marker="s", color="#d62728", label="Energy")
        ax.set_title(title)
        ax.set_xlabel(prefix)
        ax.set_ylabel("Cost (%)")
        ax2.set_ylabel("mJ/device/slot")
        if prefix == "d":
            ax.plot(x, [100 * value for value in x], color="black", linestyle="--", linewidth=.8,
                    label="Budget target d")
        else:
            ax.axhline(10, color="black", linestyle="--", linewidth=.7)
        ax.grid(alpha=.2)
    save(fig, "Fig10_sensitivity")


def failure_components():
    data = rows("evidence5_summary.csv", SUMMARY)
    order = [("split_exact", "SFR-SAC"), ("penalty_beta5", "Penalty SAC"),
             ("td3", "Safe TD3"), ("dpp_rule", "Lyapunov-inspired rule"), ("vanilla", "Vanilla SAC")]
    selected = [next(r for r in data if r["experiment"] == "main" and r["variant"] == variant and r["N"] == "70")
                for variant, _ in order]
    xlabels = [label for _, label in order]
    labels = ["Local computing", "Edge delivery", "MEC service"]
    fields = ["local_failure_percent_mean", "edge_delivery_failure_percent_mean", "mec_service_failure_percent_mean"]
    values = [[float(row[field]) for row in selected] for field in fields]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    bottom = np.zeros(len(selected))
    for label, series, color in zip(labels, values, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        ax.bar(xlabels, series, bottom=bottom, label=label, color=color)
        bottom += np.asarray(series)
    ax.set_ylabel("Failure contribution (%)")
    ax.set_title("Service-failure composition at N=70")
    ax.tick_params(axis="x", rotation=18)
    ax.legend(frameon=False)
    save(fig, "Fig11_failure_components")


def contact_sheet():
    names = [f"Fig{i}" for i in range(1, 12)]
    files = [FIGURES / f"{name}_{suffix}.png" for name, suffix in (
        ("Fig1", "main_performance"), ("Fig2", "fair_critics"), ("Fig3", "penalty_form"),
        ("Fig4", "per_slot"), ("Fig5", "ap_association"), ("Fig6", "model_mismatch"),
        ("Fig7", "cpo"), ("Fig8", "horizon"), ("Fig9", "latency_scaling"),
        ("Fig10", "sensitivity"), ("Fig11", "failure_components"))]
    images = [plt.imread(path) for path in files]
    fig, axes = plt.subplots(4, 3, figsize=(12, 13))
    for ax, image, name in zip(axes.flat, images, names):
        ax.imshow(image)
        ax.set_title(name)
        ax.axis("off")
    for ax in axes.flat[len(images):]:
        ax.axis("off")
    fig.suptitle("SFR-SAC Paper1 final figure contact sheet", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(FIGURES / "Figures_contact_sheet.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "Figures_contact_sheet.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    main_results()
    fair_critics()
    penalty_form()
    per_slot()
    ap_ablation()
    mismatch()
    cpo()
    horizon()
    latency()
    sensitivity()
    failure_components()
    contact_sheet()
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
