"""Evaluation metrics shared by learned and non-learning policies."""
from __future__ import annotations

import numpy as np


class EvalAccumulator:
    def __init__(self, n, budget, task_eps=0.01):
        self.n, self.budget, self.eps = n, float(budget), task_eps
        self.rows, self.episode = [], []
        self.fail = np.zeros(n)
        self.bits = np.zeros(n)
        self.task_fail = 0
        self.slots = 0
        self.slot_costs = []
        self.best_ap_hits = 0
        self.load_cv = []

    def add(self, info):
        c = float(info["service_failure_ratio"])
        self.slot_costs.append(c)
        f, a = info["failed_bits_vec"], info["task_bits_vec"]
        self.fail += f
        self.bits += a
        self.task_fail += int(np.sum(f > self.eps * a))
        self.slots += 1
        if "ap_index_vec" in info:
            self.best_ap_hits += int(np.sum(info["ap_index_vec"] == info["best_ap_vec"]))
            counts = info["ap_counts"]
            if counts.sum() > 0:
                self.load_cv.append(float(np.std(counts) / max(np.mean(counts), 1e-9)))
        self.episode.append(
            {
                "cost_percent": 100 * c,
                "energy_mJ_device_slot": 1000 * float(info["energy"]),
                "bfsr_percent": 100.0 * float(c <= self.budget + 1e-12),
                "local_failure_percent": 100 * float(info["local_failure_ratio"]),
                "edge_delivery_failure_percent": 100 * float(info["edge_arrival_failure_ratio"]),
                "mec_service_failure_percent": 100 * float(info["mec_admission_failure_ratio"]),
                "E_loc_mJ_device_slot": 1000 * float(info["local_energy"]) / self.n,
                "E_tx_mJ_device_slot": 1000 * float(info["tx_energy"]) / self.n,
                "avg_tx_power": float(info["tx_power_mean"]),
                "avg_offload_ratio": float(info["executed_offload_ratio_mean"]),
                "snr_db_active_mean": float(info.get("snr_db_active_mean", 0.0)),
            }
        )

    def end_episode(self):
        if self.episode:
            self.rows.append({k: float(np.mean([r[k] for r in self.episode])) for k in self.episode[0]})
        self.episode = []

    def summary(self):
        s = {k: float(np.mean([r[k] for r in self.rows])) for k in self.rows[0]}
        d = self.fail / np.maximum(self.bits, 1e-12)
        comp = 1.0 - d
        s.update(
            mean_budget_exceedance_pp=float(100 * np.maximum(np.asarray(self.slot_costs) - self.budget, 0.0).mean()),
            violation_slot_percent=float(100 * np.mean(np.asarray(self.slot_costs) > self.budget + 1e-12)),
            cost_p95_percent=float(100 * np.percentile(self.slot_costs, 95)),
            device_failure_mean_percent=float(100 * d.mean()),
            device_failure_p95_percent=float(100 * np.percentile(d, 95)),
            device_failure_max_percent=float(100 * d.max()),
            devices_above_budget=int(np.sum(d > self.budget)),
            jain_completion=float(comp.sum() ** 2 / (self.n * np.sum(comp**2))),
            task_failure_percent=float(100 * self.task_fail / (self.slots * self.n)),
            best_ap_percent=float(100 * self.best_ap_hits / (self.slots * self.n)),
            ap_load_cv=float(np.mean(self.load_cv)) if self.load_cv else 0.0,
            eval_episodes=len(self.rows),
            eval_slots=self.slots,
        )
        return s
