"""Export compact CSV and LaTeX tables from the final summary CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


LABELS = {
    "split_exact": "SFR-SAC",
    "penalty_beta5": "Penalty SAC",
    "td3": "Safe TD3",
    "dpp_rule": "Lyapunov-inspired rule",
    "vanilla": "Vanilla SAC",
    "sac_lagrangian": "SAC-Lagrangian",
    "local_first": "Local-first",
    "constant_budget": "Constant budget",
    "constant_matched_mixture": "Constant matched mixture",
    "mb_greedy_budget": "MB-greedy budget",
    "mb_greedy_matched_mixture": "MB-greedy matched mixture",
    "batch": "Batch",
    "episode": "Episode",
    "step": "Step",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def pm(row: dict[str, str], metric: str) -> str:
    return f"{number(row, metric + '_mean'):.2f} $\\pm$ {number(row, metric + '_std'):.2f}"


def write_table(path: Path, rows: list[dict[str, str]], columns: list[str], caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{'l' * len(columns)}}}",
        "\\toprule",
        " & ".join(columns) + " \\\\",
        "\\midrule",
    ]
    lines.extend([" & ".join(row[col] for col in columns) + " " + chr(92) * 2 for row in rows])
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.with_suffix(".tex").write_text("\n".join(lines), encoding="utf-8")


def metric_row(row: dict[str, str], first_key: str, first_value: str) -> dict[str, str]:
    return {
        first_key: first_value,
        "Cost (\\%)": pm(row, "cost_percent"),
        "BFSR (\\%)": pm(row, "bfsr_percent"),
        "Energy (mJ/device/slot)": pm(row, "energy_mJ_device_slot"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("results/final/data/evidence5_summary.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/final/tables"))
    args = parser.parse_args()
    rows = read_rows(args.summary)

    main_rows = [r for r in rows if r["experiment"] == "main"]
    main_rows.sort(key=lambda r: (int(r["N"]), ["split_exact", "penalty_beta5", "td3", "dpp_rule", "vanilla"].index(r["variant"])))
    table = [
        {"N": r["N"], "Algorithm": LABELS.get(r["variant"], r["variant"]),
         "Cost (\\%)": pm(r, "cost_percent"), "BFSR (\\%)": pm(r, "bfsr_percent"),
         "Energy (mJ/device/slot)": pm(r, "energy_mJ_device_slot")}
        for r in main_rows
    ]
    write_table(args.output / "main_results", table, list(table[0]), "Main comparison across system scale.", "tab:main_results")

    specs = [
        ("mechanism", "mechanism_N70", "variant", "Mechanism architecture ablation at N=70.", "tab:mechanism"),
        ("budget", "budget_sensitivity_N70", "variant", "Service-failure budget sensitivity at N=70.", "tab:budget_sensitivity"),
        ("eta", "eta_sensitivity_N70", "variant", "Service-window sensitivity at N=70.", "tab:eta_sensitivity"),
        ("dual", "lambda_update_N70", "variant", "Lambda update granularity at N=70.", "tab:lambda_update"),
        ("penalty", "penalty_beta_N70", "variant", "Fixed-penalty sensitivity at N=70.", "tab:penalty_sensitivity"),
        ("background", "background_load_N70", "variant", "MEC background-load sensitivity at N=70.", "tab:background_load"),
        ("fair_rules", "fair_rules_N70", "variant", "Matched rule baselines at N=70.", "tab:fair_rules"),
    ]
    for experiment, stem, key, caption, label in specs:
        selected = [r for r in rows if r["experiment"] == experiment]
        selected.sort(key=lambda r: r[key])
        table = [metric_row(r, "Variant", LABELS.get(r["variant"], r["variant"])) for r in selected]
        if table:
            write_table(args.output / stem, table, list(table[0]), caption, label)


if __name__ == "__main__":
    main()
