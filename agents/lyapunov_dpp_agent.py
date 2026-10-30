import numpy as np


class LyapunovDPPAgent:
    def __init__(
        self,
        num_devices: int,
        num_rsu: int = 2,
        *,
        max_buffer: float = 100000.0,
        load_scale: float = 1.0,
        slot_duration: float = 0.001,
        ul_time_fraction: float = 0.5,
        total_access_bw: float = 5000000.0,
        bs_cpu_base: float = 25000000000.0,
        local_cpu: float = 2500000000.0,
        cycles_per_bit: float = 3000.0,
        tx_power: float = 0.1,
        tx_power_control: str = "fixed",
        tx_power_min: float = 0.02,
        tx_power_max: float = 0.2,
        noise: float = 1e-07,
        local_energy_per_bit: float = 1.0,
        tx_energy_per_bit: float = 0.6,
        task_model: str = "queue",
        mode: str = "strong",
        queue_target: float = 0.52,
        max_offload: float = 1.0,
        access_margin: float = 1.0,
        edge_margin: float = 1.0,
        v_energy: float = 2.5,
        eta_delivery: float = 0.25,
        eta_queue: float = 0.25,
        delivery_limit: float = 0.06,
        queue_limit: float = 0.08,
        initial_z: float = 0.0,
    ):
        self.N = int(num_devices)
        self.R = int(num_rsu)
        self.mode = str(mode).lower()
        if self.mode not in {"strong", "standard"}:
            raise ValueError(f"Unknown LyapunovDPPAgent mode: {mode}")
        self.max_buffer = float(max_buffer)
        self.load_scale = float(load_scale)
        self.slot_duration = float(slot_duration)
        self.ul_time_fraction = float(ul_time_fraction)
        self.total_access_bw = float(total_access_bw)
        self.bs_cpu = float(bs_cpu_base)
        self.local_cpu = float(local_cpu)
        self.cycles_per_bit = float(cycles_per_bit)
        self.tx_power = float(tx_power)
        self.tx_power_control = str(tx_power_control).lower()
        if self.tx_power_control not in {"fixed", "action"}:
            raise ValueError(f"Unknown tx_power_control: {tx_power_control}")
        self.tx_power_min = float(tx_power_min)
        self.tx_power_max = float(tx_power_max)
        if not 0.0 < self.tx_power_min <= self.tx_power <= self.tx_power_max:
            raise ValueError("Require 0 < tx_power_min <= tx_power <= tx_power_max")
        self.noise = float(noise)
        self.local_energy_per_bit = float(local_energy_per_bit)
        self.tx_energy_per_bit = float(tx_energy_per_bit)
        self.task_model = str(task_model).lower()
        if self.task_model not in {"queue", "oneshot"}:
            raise ValueError(f"Unknown Lyapunov task_model: {task_model}")
        self.queue_target = float(np.clip(queue_target, 0.05, 0.79))
        self.max_offload = float(np.clip(max_offload, 0.0, 1.0))
        self.access_margin = float(max(access_margin, 0.05))
        self.edge_margin = float(max(edge_margin, 0.05))
        self.v_energy = float(max(v_energy, 0.0))
        self.eta_delivery = float(max(eta_delivery, 0.0))
        self.eta_queue = float(max(eta_queue, 0.0))
        self.delivery_limit = float(max(delivery_limit, 0.0))
        self.queue_limit = float(max(queue_limit, 0.0))
        self.z_delivery = float(max(initial_z, 0.0))
        self.z_queue = float(max(initial_z, 0.0))
        self.lambda_val = self.z_delivery + self.z_queue
        self.alpha = 0.0

    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _ratio_to_action(ratio):
        return np.clip(2.0 * ratio - 1.0, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _rel_to_action(rel):
        rel = np.clip(rel, 1e-06, 1.0 - 1e-06)
        return np.clip(np.log(rel / (1.0 - rel)) / 4.0, -1.0, 1.0).astype(np.float32)

    def _ap_indices_to_action(self, idx):
        if self.R <= 1:
            return np.zeros(self.N, dtype=np.float32)
        return (-1.0 + 2.0 * np.asarray(idx, dtype=np.float32) / (self.R - 1)).astype(np.float32)

    def _tx_power_to_action(self, tx_power):
        tx_power = np.asarray(tx_power, dtype=np.float32)
        p01 = (tx_power - self.tx_power_min) / (self.tx_power_max - self.tx_power_min + 1e-12)
        return np.clip(2.0 * p01 - 1.0, -1.0, 1.0).astype(np.float32)

    def _power_action(self, offload_ratio, h_norm):
        if self.tx_power_control != "action":
            return np.zeros(0, dtype=np.float32)
        delivery_pressure = self.z_delivery / (1.0 + self.z_delivery)
        p01 = (
            0.18
            + 0.42 * delivery_pressure
            + 0.22 * np.clip(offload_ratio, 0.0, 1.0)
            + 0.18 * (1.0 - np.clip(h_norm, 0.0, 1.0))
        )
        p01 = np.clip(p01, 0.0, 1.0)
        tx_power = self.tx_power_min + p01 * (self.tx_power_max - self.tx_power_min)
        return self._tx_power_to_action(tx_power)

    def _parse_state(self, state):
        s = np.asarray(state, dtype=np.float32).flatten()
        n = self.N
        r = self.R
        if s.shape[0] < 3 * n:
            raise ValueError(f"State dim too small: got {s.shape[0]}, expected >= {3 * n}")
        h_norm = np.clip(s[:n], 0.0, 1.0)
        q_norm = np.clip(s[n : 2 * n], 0.0, 1.0)
        profile = np.clip(s[2 * n : 3 * n], 0.0, 1.0)
        start = 3 * n
        ap_load = s[start : start + r] if s.shape[0] >= start + r else np.zeros(r, dtype=np.float32)
        ap_load = np.clip(ap_load, 0.0, 1.0).astype(np.float32)
        return (h_norm, q_norm, profile, ap_load)

    def select_action(self, state, deterministic: bool = False):
        h_norm, q_norm, profile, ap_load = self._parse_state(state)
        return self._select_standard_action(h_norm, q_norm, profile, ap_load)

    def _select_standard_action(self, h_norm, q_norm, profile, ap_load):
        delivery_pressure = self.z_delivery / (1.0 + self.z_delivery)
        queue_pressure = self.z_queue / (1.0 + self.z_queue)
        queue_drive = self._sigmoid(10.0 * (q_norm - self.queue_target))
        channel_drive = self._sigmoid(6.0 * (h_norm - 0.45))
        energy_drive = self._sigmoid(4.0 * (profile - 0.45)) * channel_drive
        pressure_scale = 1.0 + 0.55 * queue_pressure - 0.45 * delivery_pressure
        pressure_scale = float(np.clip(pressure_scale, 0.45, 1.3))
        continuous = (
            self.max_offload
            * (0.68 * queue_drive + 0.32 * self.v_energy * energy_drive)
            * pressure_scale
        )
        continuous = np.clip(continuous, 0.0, self.max_offload)
        levels = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        levels = levels[levels <= self.max_offload + 1e-09]
        if levels.size == 0:
            levels = np.array([0.0], dtype=np.float32)
        idx_level = np.abs(continuous[:, None] - levels[None, :]).argmin(axis=1)
        chosen = levels[idx_level].astype(np.float32)
        if self.R <= 1:
            rsu_idx = np.zeros(self.N, dtype=np.int64)
        else:
            active = chosen > 1e-06
            virtual_load = ap_load.astype(np.float32).copy() * max(float(self.N), 1.0)
            rsu_idx = np.zeros(self.N, dtype=np.int64)
            for dev in np.flatnonzero(active):
                chosen_ap = int(np.argmin(virtual_load))
                rsu_idx[dev] = chosen_ap
                virtual_load[chosen_ap] += 1.0
        rel = np.full(self.N, 0.05, dtype=np.float32)
        action = np.concatenate(
            [
                self._ratio_to_action(chosen),
                self._rel_to_action(rel),
                self._ap_indices_to_action(rsu_idx),
                self._power_action(chosen, h_norm),
            ]
        ).astype(np.float32)
        return action

    def update_virtual_queue(self, cost, info=None):
        info = info or {}
        delivery = float(
            info.get("service_failure_ratio", info.get("offloading_delivery_failure_ratio", cost))
        )
        queue = float(info.get("queue_violation", 0.0))
        self.z_delivery = max(
            0.0, self.z_delivery + self.eta_delivery * (delivery - self.delivery_limit)
        )
        self.z_queue = max(0.0, self.z_queue + self.eta_queue * (queue - self.queue_limit))
        self.lambda_val = self.z_delivery + self.z_queue

    def update(self, *args, **kwargs):
        return {"actor_loss": 0.0, "loss_q1": 0.0, "loss_q2": 0.0, "loss_qc": 0.0}
