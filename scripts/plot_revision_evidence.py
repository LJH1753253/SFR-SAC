"""Generate the primary-evidence figures for the revision archive."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "revision_20261108"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"


def read_csv(name: str):
    with (TABLES / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def mean_sd(value: str):
    match = re.match(r"\s*([0-9.]+)\s*[±+/-]+\s*([0-9.]+)", value)
    if not match:
        return float(value), 0.0
    return float(match.group(1)), float(match.group(2))


def save(fig, name: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)


def fair_critics():
    wanted = ["sfr", "td3", "td3_struct", "td3_struct_split"]
    rows = {row["Variant"]: row for row in read_csv("fair_critics_full5.csv")}
    labels = ["SFR-SAC", "Safe TD3", "TD3 +\nStructured Critics", "TD3 +\nStructured Critics\n+ Split Actor"]
    colors = ["#1f77b4", "#7f7f7f", "#d62728", "#ff7f0e"]
    metrics = [("Failure (%)", "Failure (%)"), ("Energy (mJ)", "Energy (mJ/device/slot)"), ("BFSR (%)", "BFSR (%)")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
    x = np.arange(len(wanted))
    for ax, (column, title) in zip(axes, metrics):
        means, errors = zip(*(mean_sd(rows[key][column]) for key in wanted))
        ax.bar(x, means, yerr=errors, capsize=3, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=20, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].axhline(10, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Percent")
    axes[1].set_ylabel("mJ/device/slot")
    axes[2].set_ylabel("Percent")
    save(fig, "Fig_revision_fair_critics.pdf")


def ap_ablation():
    wanted = ["sfr", "sfr_maxgain", "sfr_apind"]
    rows = {row["Variant"]: row for row in read_csv("ap.csv")}
    labels = ["AP-dependent\nlearned AP", "Best-channel\nAP", "AP-independent\nchannel"]
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    x = np.arange(len(wanted))
    for ax, column, title, ylabel in zip(
        axes,
        ("Failure (%)", "Energy (mJ)"),
        ("AP/channel ablation: service failure", "AP/channel ablation: energy"),
        ("Percent", "mJ/device/slot"),
    ):
        means, errors = zip(*(mean_sd(rows[key][column]) for key in wanted))
        ax.bar(x, means, yerr=errors, capsize=3, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, labels, rotation=18, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].axhline(10, color="black", linestyle="--", linewidth=0.8)
    save(fig, "Fig_revision_ap_channel_ablation.pdf")


def latency():
    rows = read_csv("latency.csv")
    selected = [row for row in rows if row["Policy"] == "sfr"]
    selected.sort(key=lambda row: (row["Weights"] != "trained", int(row["N"])))
    labels = [f"N={row['N']}" + (" trained" if row["Weights"] == "trained" else "") for row in selected]
    x = np.arange(len(selected))
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    for column, marker in (("Median (ms)", "o"), ("P95 (ms)", "s"), ("P99 (ms)", "^") ):
        ax.plot(x, [float(row[column]) for row in selected], marker=marker, linewidth=1.8, label=column)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Policy/action-decode time (ms)")
    ax.set_title("SFR-SAC online decision latency")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save(fig, "Fig_revision_latency.pdf")


if __name__ == "__main__":
    fair_critics()
    ap_ablation()
    latency()
    print(f"Wrote figures to {FIGURES}")
