import numpy as np
import gymnasium as gym
from gymnasium import spaces
from agents.action_projection import a_ap_to_rsu_index


class VecOffloadingEnv(gym.Env):
    """One-slot workload service with continuous control and categorical AP selection."""

    def __init__(
        self,
        num_devices=70,
        delay_budget=0.03,
        delay_budget_low_factor=1.0,
        delay_budget_high_factor=1.0,
        reward_mode="energy",
        reward_delay_mode="normalized",
        task_model="oneshot",
        local_energy_model="physical_joule_cpu",
        local_energy_per_bit=1.0,
        local_energy_zeta_bar=1e-28,
        local_cycles_per_bit_ref=None,
        tx_energy_per_bit=0.45,
        tx_energy_model="physical_pa_circuit",
        tx_energy_rate_sensitivity=1.0,
        tx_energy_rate_min_multiplier=0.5,
        tx_energy_rate_max_multiplier=3.0,
        tx_energy_ref_rate_mode="nominal",
        tx_energy_time_scale=1.0,
        tx_pa_efficiency=0.35,
        tx_circuit_power=0.05,
        tx_power_control="action",
        tx_power_base=0.1,
        tx_power_min=0.02,
        tx_power_max=0.2,
        queue_holding_energy_per_bit=0.0,
        energy_heterogeneity_span=0.3,
        energy_ref=1000000.0,
        max_ep_steps=200,
        load_scale=22.0,
        max_buffer=100000.0,
        cost_delay_coef=0.0,
        initial_queue_ratio=0.13,
        initial_queue_profile="uniform",
        initial_queue_heterogeneity_span=0.0,
        initial_queue_jitter=0.0,
        cost_outage_coef=1.0,
        cost_queue_coef=0.0,
        k_cong=0.7,
        max_episodes=1600,
        requeue_on_outage=True,
        redundancy_sigmoid_k=4.0,
        redundancy_power=2.5,
        rel_cong_coef=0.7,
        redundancy_delay_factor=0.25,
        p_out_max=0.6,
        link_failure_model="shortfall",
        ul_time_fraction=0.3,
        slot_duration=0.03,
        shortfall_projection_margin=0.0,
        shortfall_projection_scope="access",
        num_rsu=6,
        backhaul_delay=0.002,
        backhaul_bw=1000000000.0,
        total_access_bw=7800000.0,
        bs_cpu_base=80000000000.0,
        channel_dynamics="slow",
        channel_corr=0.92,
        channel_noise_std=0.04,
        mec_resource_dynamics="slow",
        mec_resource_corr=0.95,
        mec_resource_noise_std=0.04,
        mec_resource_min=0.7,
        mec_resource_max=1.1,
        mec_background_load_base=0.05,
        mec_background_load_min=0.0,
        mec_background_load_max=0.3,
        oneshot_cost_components="full",
        risk_ablation="delay_outage",
        risk_cost_mode="impl_full",
        action_ablation="no_rel",
        action_mode="hybrid_ap_gumbel",
        ap_projection_mode="legacy",
        tx_energy_accounting="delivered",
    ):
        super().__init__()
        if tx_energy_accounting not in {"delivered", "attempted"}:
            raise ValueError("tx_energy_accounting must be delivered or attempted")
        self.tx_energy_accounting = tx_energy_accounting
        for key, value, expected in [
            ("task_model", task_model, "oneshot"),
            ("reward_mode", reward_mode, "energy"),
            ("action_mode", action_mode, "hybrid_ap_gumbel"),
            ("local_energy_model", local_energy_model, "physical_joule_cpu"),
            ("tx_energy_model", tx_energy_model, "physical_pa_circuit"),
            ("tx_power_control", tx_power_control, "action"),
            ("oneshot_cost_components", oneshot_cost_components, "full"),
        ]:
            if value != expected:
                raise ValueError(f"This release requires {key}={expected}")
        self.N = int(num_devices)
        self.R = int(num_rsu)
        self.initial_queue_ratio = float(initial_queue_ratio)
        self.initial_queue_profile = str(initial_queue_profile).lower()
        self.initial_queue_heterogeneity_span = float(initial_queue_heterogeneity_span)
        self.initial_queue_jitter = float(initial_queue_jitter)
        allowed_queue_profiles = {"uniform", "tiered"}
        if self.initial_queue_profile not in allowed_queue_profiles:
            raise ValueError(f"Unknown initial_queue_profile: {initial_queue_profile}")
        if self.initial_queue_heterogeneity_span < 0.0:
            raise ValueError("initial_queue_heterogeneity_span must be non-negative")
        if self.initial_queue_jitter < 0.0:
            raise ValueError("initial_queue_jitter must be non-negative")
        self.N_ref = 2
        self.capacity_beta = 1.0
        self.buffer_beta = 0.6
        self.capacity_scale = 1.0
        self.buffer_scale = 1.0
        self.cost_outage_coef = float(cost_outage_coef)
        self.cost_queue_coef = float(cost_queue_coef)
        self.cost_delay_coef = float(cost_delay_coef)
        self.max_episodes = int(max_episodes)
        self.current_episode = 0
        self.noise = 1e-07
        self.tx_power_base = float(tx_power_base)
        self.tx_power_min = float(tx_power_min)
        self.tx_power_max = float(tx_power_max)
        self.tx_power_control = str(tx_power_control).lower()
        allowed_tx_power_control = {"fixed", "action"}
        if self.tx_power_control not in allowed_tx_power_control:
            raise ValueError(f"Unknown tx_power_control: {tx_power_control}")
        if self.tx_power_base <= 0.0:
            raise ValueError("tx_power_base must be positive")
        if not 0.0 < self.tx_power_min <= self.tx_power_max:
            raise ValueError("Require 0 < tx_power_min <= tx_power_max")
        self.tx_power = self.tx_power_base
        self.tx_power_vec = np.full(self.N, self.tx_power, dtype=np.float32)
        self.last_tx_power_mean = float(self.tx_power)
        total_access_bw_scaled = float(total_access_bw) * self.capacity_scale
        if self.R <= 0:
            raise ValueError("num_rsu must be >= 1")
        self.rsu_access_bw = np.ones(self.R, dtype=np.float32) * (total_access_bw_scaled / self.R)
        self.backhaul_delay = float(backhaul_delay)
        self.backhaul_bw = float(backhaul_bw)
        self.bs_cpu_nominal = float(bs_cpu_base) * self.capacity_scale
        self.bs_cpu = self.bs_cpu_nominal
        self.local_cpu = 2500000000.0
        self.cycles_per_bit = 3000.0
        self.max_buffer = float(max_buffer) * self.buffer_scale
        if self.max_buffer <= 0.0:
            raise ValueError("max_buffer must be positive")
        self.arrival_low_base = 800
        self.arrival_high_base = 2000
        self.load_scale = float(load_scale)
        self.task_cycles_low_factor = 0.8
        self.task_cycles_high_factor = 1.2
        self.queue_threshold = 0.8
        self.delay_budget = float(delay_budget)
        self.delay_budget_low_factor = float(delay_budget_low_factor)
        self.delay_budget_high_factor = float(delay_budget_high_factor)
        self.reward_mode = str(reward_mode).lower()
        self.task_model = str(task_model).lower()
        allowed_task_model = {"queue", "oneshot"}
        if self.task_model not in allowed_task_model:
            raise ValueError(f"Unknown task_model: {task_model}")
        allowed_reward_mode = {"delay", "energy"}
        if self.reward_mode not in allowed_reward_mode:
            raise ValueError(f"Unknown reward_mode: {reward_mode}")
        self.reward_delay_mode = str(reward_delay_mode).lower()
        allowed_reward_delay = {"average", "normalized"}
        if self.reward_delay_mode not in allowed_reward_delay:
            raise ValueError(f"Unknown reward_delay_mode: {reward_delay_mode}")
        self.local_energy_model = str(local_energy_model).lower()
        self.local_energy_per_bit = float(local_energy_per_bit)
        self.local_cycles_per_bit_ref = (
            float(local_cycles_per_bit_ref)
            if local_cycles_per_bit_ref is not None
            else float(self.cycles_per_bit)
        )
        if local_energy_zeta_bar is not None:
            self.local_energy_zeta_bar = float(local_energy_zeta_bar)
        else:
            self.local_energy_zeta_bar = 1e-28
        self.tx_energy_per_bit = float(tx_energy_per_bit)
        self.tx_energy_model = str(tx_energy_model).lower()
        self.tx_energy_rate_sensitivity = float(tx_energy_rate_sensitivity)
        self.tx_energy_rate_min_multiplier = float(tx_energy_rate_min_multiplier)
        self.tx_energy_rate_max_multiplier = float(tx_energy_rate_max_multiplier)
        self.tx_energy_ref_rate_mode = str(tx_energy_ref_rate_mode).lower()
        self.tx_energy_time_scale = float(tx_energy_time_scale)
        self.tx_pa_efficiency = float(tx_pa_efficiency)
        self.tx_circuit_power = float(tx_circuit_power)
        self.queue_holding_energy_per_bit = float(queue_holding_energy_per_bit)
        self.energy_heterogeneity_span = float(energy_heterogeneity_span)
        self.energy_ref = float(energy_ref)
        if self.energy_ref <= 0.0:
            raise ValueError("energy_ref must be positive")
        if (
            self.local_energy_per_bit < 0.0
            or self.tx_energy_per_bit < 0.0
            or self.queue_holding_energy_per_bit < 0.0
        ):
            raise ValueError("energy coefficients must be non-negative")
        if self.energy_heterogeneity_span < 0.0:
            raise ValueError("energy_heterogeneity_span must be non-negative")
        allowed_local_energy_model = {"effective_old", "physical_fixed_cpu", "physical_joule_cpu"}
        allowed_tx_energy_model = {
            "per_bit",
            "rate_scaled",
            "fixed_power_time",
            "physical_pa_circuit",
        }
        allowed_tx_ref_rate_mode = {"nominal", "median"}
        if self.local_energy_model not in allowed_local_energy_model:
            raise ValueError(f"Unknown local_energy_model: {local_energy_model}")
        if self.local_cycles_per_bit_ref <= 0.0:
            raise ValueError("local_cycles_per_bit_ref must be positive")
        if self.local_energy_zeta_bar < 0.0:
            raise ValueError("local_energy_zeta_bar must be non-negative")
        if self.tx_energy_model not in allowed_tx_energy_model:
            raise ValueError(f"Unknown tx_energy_model: {tx_energy_model}")
        if self.tx_energy_ref_rate_mode not in allowed_tx_ref_rate_mode:
            raise ValueError(f"Unknown tx_energy_ref_rate_mode: {tx_energy_ref_rate_mode}")
        if self.tx_energy_rate_sensitivity < 0.0:
            raise ValueError("tx_energy_rate_sensitivity must be non-negative")
        if self.tx_energy_time_scale <= 0.0:
            raise ValueError("tx_energy_time_scale must be positive")
        if not 0.0 < self.tx_pa_efficiency <= 1.0:
            raise ValueError("tx_pa_efficiency must be in (0, 1]")
        if self.tx_circuit_power < 0.0:
            raise ValueError("tx_circuit_power must be non-negative")
        if not 0.0 < self.tx_energy_rate_min_multiplier <= self.tx_energy_rate_max_multiplier:
            raise ValueError(
                "Require 0 < tx_energy_rate_min_multiplier <= tx_energy_rate_max_multiplier"
            )
        if self.delay_budget <= 0.0:
            raise ValueError("delay_budget must be positive")
        if self.delay_budget_low_factor <= 0.0 or self.delay_budget_high_factor <= 0.0:
            raise ValueError("delay_budget factors must be positive")
        if self.delay_budget_high_factor < self.delay_budget_low_factor:
            raise ValueError("delay_budget_high_factor must be >= delay_budget_low_factor")
        if abs(self.delay_budget_high_factor - self.delay_budget_low_factor) < 1e-12:
            delay_budget_factors = np.full(self.N, self.delay_budget_low_factor, dtype=np.float32)
        else:
            delay_budget_factors = np.random.uniform(
                self.delay_budget_low_factor, self.delay_budget_high_factor, size=self.N
            ).astype(np.float32)
        self.delay_budget_vec = (self.delay_budget * delay_budget_factors).astype(np.float32)
        factor_span = self.delay_budget_high_factor - self.delay_budget_low_factor
        if factor_span < 1e-12:
            self.delay_budget_state = np.ones(self.N, dtype=np.float32)
        else:
            self.delay_budget_state = (
                (delay_budget_factors - self.delay_budget_low_factor) / (factor_span + 1e-12)
            ).astype(np.float32)
        self.energy_factor = self._make_device_profile(self.energy_heterogeneity_span, shuffle=True)
        self.local_energy_per_bit_vec = (self.local_energy_per_bit * self.energy_factor).astype(
            np.float32
        )
        self.tx_energy_per_bit_vec = (self.tx_energy_per_bit * self.energy_factor).astype(
            np.float32
        )
        self.queue_holding_energy_per_bit_vec = (
            self.queue_holding_energy_per_bit * self.energy_factor
        ).astype(np.float32)
        self.device_profile_state = self._normalize_profile_state(self.energy_factor)
        self.delay_norm = 0.06
        self.slot_duration = float(slot_duration)
        self.ul_time_fraction = float(ul_time_fraction)
        self.shortfall_projection_margin = float(shortfall_projection_margin)
        self.shortfall_projection_scope = str(shortfall_projection_scope).lower()
        self.snr_threshold_linear = 10.0
        self.channel_dynamics = str(channel_dynamics).lower()
        self.channel_corr = float(channel_corr)
        self.channel_noise_std = float(channel_noise_std)
        self.channel_min = 0.5
        self.channel_max = 1.0
        self.tx_energy_ref_rate = self._compute_nominal_uplink_rate()
        self.last_tx_energy_ref_rate = float(self.tx_energy_ref_rate)
        allowed_channel_dynamics = {"iid", "slow"}
        if self.channel_dynamics not in allowed_channel_dynamics:
            raise ValueError(f"Unknown channel_dynamics: {channel_dynamics}")
        if not 0.0 <= self.channel_corr < 1.0:
            raise ValueError("channel_corr must be in [0, 1)")
        if self.channel_noise_std < 0.0:
            raise ValueError("channel_noise_std must be non-negative")
        self.mec_resource_dynamics = str(mec_resource_dynamics).lower()
        self.mec_resource_corr = float(mec_resource_corr)
        self.mec_resource_noise_std = float(mec_resource_noise_std)
        self.mec_resource_min = float(mec_resource_min)
        self.mec_resource_max = float(mec_resource_max)
        allowed_mec_dynamics = {"fixed", "slow"}
        if self.mec_resource_dynamics not in allowed_mec_dynamics:
            raise ValueError(f"Unknown mec_resource_dynamics: {mec_resource_dynamics}")
        if not 0.0 <= self.mec_resource_corr < 1.0:
            raise ValueError("mec_resource_corr must be in [0, 1)")
        if self.mec_resource_noise_std < 0.0:
            raise ValueError("mec_resource_noise_std must be non-negative")
        if self.mec_resource_min <= 0.0 or self.mec_resource_max < self.mec_resource_min:
            raise ValueError("Require 0 < mec_resource_min <= mec_resource_max")
        old_resource_ratio_min_eff = float(self.mec_resource_min)
        old_resource_ratio_max_eff = min(float(self.mec_resource_max), 1.0)
        if (
            old_resource_ratio_min_eff <= 0.0
            or old_resource_ratio_max_eff < old_resource_ratio_min_eff
        ):
            raise ValueError(
                "Require 0 < effective mec resource min <= effective mec resource max <= 1"
            )
        derived_b_min = max(0.0, 1.0 - old_resource_ratio_max_eff)
        derived_b_max = min(0.95, 1.0 - old_resource_ratio_min_eff)
        self.mec_background_load_min = (
            derived_b_min if mec_background_load_min is None else float(mec_background_load_min)
        )
        self.mec_background_load_max = (
            derived_b_max if mec_background_load_max is None else float(mec_background_load_max)
        )
        if not 0.0 <= self.mec_background_load_min <= self.mec_background_load_max < 1.0:
            raise ValueError("Require 0 <= mec_background_load_min <= mec_background_load_max < 1")
        if mec_background_load_base is None:
            self.mec_background_load_base = 0.5 * (
                self.mec_background_load_min + self.mec_background_load_max
            )
        else:
            self.mec_background_load_base = float(mec_background_load_base)
        if (
            not self.mec_background_load_min
            <= self.mec_background_load_base
            <= self.mec_background_load_max
        ):
            raise ValueError("Require mec_background_load_min <= base <= max")
        self.mec_resource_ratio_min_eff = 1.0 - self.mec_background_load_max
        self.mec_resource_ratio_max_eff = 1.0 - self.mec_background_load_min
        if not (
            self.mec_resource_ratio_min_eff > 0.0
            and self.mec_resource_ratio_max_eff <= 1.0
            and (self.mec_resource_ratio_min_eff <= self.mec_resource_ratio_max_eff)
        ):
            raise ValueError("Invalid effective MEC resource ratio range")
        self.mec_background_load_ratio = 0.0
        self.mec_resource_ratio = 1.0
        self.redundancy_delay_factor = float(redundancy_delay_factor)
        self.p_fast = 0.2
        self.p_safe = 0.015
        self.k_cong = float(k_cong)
        self.cong_power = 2.0
        self.requeue_on_outage = bool(requeue_on_outage)
        self.action_mode = str(action_mode).lower()
        allowed_action_mode = {"legacy", "hybrid_ap_gumbel", "hybrid_ap_onehot"}
        if self.action_mode not in allowed_action_mode:
            raise ValueError(f"Unknown action_mode: {action_mode}")
        self.hybrid_ap_enabled = self.action_mode in {"hybrid_ap_gumbel", "hybrid_ap_onehot"}
        if self.hybrid_ap_enabled and self.tx_power_control != "action":
            raise ValueError("hybrid AP action mode requires tx_power_control='action'")
        self.ap_projection_mode = str(ap_projection_mode).lower()
        self.link_failure_model = str(link_failure_model).lower()
        self.redundancy_sigmoid_k = float(redundancy_sigmoid_k)
        self.redundancy_power = float(redundancy_power)
        self.rel_cong_coef = float(rel_cong_coef)
        self.p_out_max = float(p_out_max)
        allowed_risk = {"full", "delay_only", "delay_outage"}
        allowed_risk_cost = {"impl_full", "paper_eq29"}
        allowed_action = {"full", "no_rel", "no_ap"}
        allowed_link_failure = {"outage", "shortfall"}
        allowed_projection_scope = {"access", "delivery"}
        allowed_oneshot_cost_components = {
            "full",
            "no_mec_admission",
            "no_local",
            "edge_arrival_only",
            "mec_admission_only",
        }
        if risk_ablation not in allowed_risk:
            raise ValueError(f"Unknown risk_ablation: {risk_ablation}")
        if risk_cost_mode not in allowed_risk_cost:
            raise ValueError(f"Unknown risk_cost_mode: {risk_cost_mode}")
        if action_ablation not in allowed_action:
            raise ValueError(f"Unknown action_ablation: {action_ablation}")
        if self.link_failure_model not in allowed_link_failure:
            raise ValueError(f"Unknown link_failure_model: {link_failure_model}")
        if self.shortfall_projection_scope not in allowed_projection_scope:
            raise ValueError(f"Unknown shortfall_projection_scope: {shortfall_projection_scope}")
        self.oneshot_cost_components = str(oneshot_cost_components).lower()
        if self.oneshot_cost_components not in allowed_oneshot_cost_components:
            raise ValueError(f"Unknown oneshot_cost_components: {oneshot_cost_components}")
        self.risk_ablation = risk_ablation
        self.risk_cost_mode = risk_cost_mode
        self.action_ablation = action_ablation
        self.queues = np.zeros(self.N, dtype=np.float32)
        self.initial_queue_ratio_vec = np.zeros(self.N, dtype=np.float32)
        self.h = np.random.uniform(0.2, 1.0, size=self.N).astype(np.float32)
        self.current_task_bits = np.zeros(self.N, dtype=np.float32)
        self.current_task_cycles_per_bit = np.full(self.N, self.cycles_per_bit, dtype=np.float32)
        self.current_task_deadline = self.delay_budget_vec.copy().astype(np.float32)
        self.current_task_profile_state = np.zeros(self.N, dtype=np.float32)
        self.last_total_delay = 0.0
        self.last_avg_delay = 0.0
        self.last_delay_norm_ratio = 0.0
        self.last_delay_margin = 0.0
        self.last_queue_margin = 0.0
        self.last_snr_margin = 0.0
        self.last_rsu_load_norm = np.zeros(self.R, dtype=np.float32)
        self.last_service_failure = 0.0
        self.last_energy_norm = 0.0
        self.last_offload_ratio = 0.0
        self.last_ap_entropy = 0.0
        self.last_ap_max_prob = 0.0
        self.last_ap_load_distribution = np.zeros(self.R, dtype=np.float32)
        self.state_dim = self.N * 3 + self.R + 8
        self.power_action_enabled = self.tx_power_control == "action"
        self.action_dim = 2 * self.N + self.N * self.R
        low = np.zeros(self.state_dim, dtype=np.float32)
        low[-3:] = -1.0
        high = np.ones(self.state_dim, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low, high=high, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        self.step_count = 0
        self.ep_step_count = 0
        self.max_ep_steps = int(max_ep_steps)

    def update_training_progress(self, episode):
        self.current_episode = int(episode)

    def _make_device_profile(self, span, shuffle=False):
        span = float(span)
        if span <= 1e-12 or self.N <= 1:
            return np.ones(self.N, dtype=np.float32)
        low = max(1.0 - span, 0.001)
        high = 1.0 + span
        profile = np.linspace(low, high, self.N, dtype=np.float32)
        profile = profile / (float(np.mean(profile)) + 1e-12)
        if shuffle:
            profile = profile.copy()
            np.random.shuffle(profile)
        return profile.astype(np.float32)

    @staticmethod
    def _normalize_profile_state(profile):
        profile = np.asarray(profile, dtype=np.float32)
        p_min = float(np.min(profile))
        p_max = float(np.max(profile))
        if p_max - p_min < 1e-12:
            return np.ones_like(profile, dtype=np.float32)
        return ((profile - p_min) / (p_max - p_min + 1e-12)).astype(np.float32)

    def _compute_nominal_uplink_rate(self):
        mean_h = 0.5 * (self.channel_min + self.channel_max)
        nominal_snr = self.tx_power_base * mean_h / (self.noise + 1e-12)
        nominal_bw = float(np.mean(self.rsu_access_bw))
        return float(max(nominal_bw * np.log2(1.0 + nominal_snr), 1e-09))

    def _compute_local_energy(self, success_bits, cycles_per_bit=None):
        success_bits = np.asarray(success_bits, dtype=np.float32)
        if cycles_per_bit is None:
            cycles_per_bit = np.full(self.N, self.cycles_per_bit, dtype=np.float32)
        cycles_per_bit = np.asarray(cycles_per_bit, dtype=np.float32)
        if cycles_per_bit.size == 1:
            cycles_per_bit = np.full(self.N, float(cycles_per_bit.reshape(-1)[0]), dtype=np.float32)
        return float(
            np.sum(
                self.local_energy_zeta_bar
                * self.energy_factor
                * cycles_per_bit
                * success_bits
                * float(self.local_cpu) ** 2
            )
        )

    def _decode_tx_power(self, action_tail=None):
        if not self.power_action_enabled:
            tx_power_vec = np.full(self.N, self.tx_power, dtype=np.float32)
        else:
            if action_tail is None or len(action_tail) < self.N:
                raise ValueError(
                    "Transmit-power action is enabled but missing the power action segment"
                )
            a_power = np.asarray(action_tail[: self.N], dtype=np.float32)
            power01 = np.clip((a_power + 1.0) / 2.0, 0.0, 1.0)
            tx_power_vec = (
                self.tx_power_min + power01 * (self.tx_power_max - self.tx_power_min)
            ).astype(np.float32)
        self.tx_power_vec = tx_power_vec.astype(np.float32)
        self.last_tx_power_mean = float(np.mean(self.tx_power_vec))
        return self.tx_power_vec

    def _compute_tx_energy(self, requested_edge, access_rate, tx_power_vec=None):
        requested_edge = np.asarray(requested_edge, dtype=np.float32)
        access_rate = np.asarray(access_rate, dtype=np.float32)
        if tx_power_vec is None:
            tx_power_vec = np.full(self.N, self.tx_power, dtype=np.float32)
        tx_power_vec = np.asarray(tx_power_vec, dtype=np.float32)
        tx_time_for_energy = requested_edge / (access_rate + 1e-09)
        tx_power_draw = tx_power_vec / self.tx_pa_efficiency + self.tx_circuit_power
        return float(np.sum(self.energy_factor * tx_power_draw * tx_time_for_energy))

    def _reset_channel_state(self):
        self.h = np.random.uniform(self.channel_min, self.channel_max, size=self.N).astype(
            np.float32
        )

    def _advance_channel_state(self):
        if self.channel_dynamics == "iid":
            self._reset_channel_state()
            return
        rho = float(self.channel_corr)
        channel_mean = 0.5 * (self.channel_min + self.channel_max)
        innovation_scale = self.channel_noise_std * np.sqrt(max(1.0 - rho * rho, 0.0))
        eps = np.random.normal(0.0, 1.0, size=self.N).astype(np.float32)
        self.h = (channel_mean + rho * (self.h - channel_mean) + innovation_scale * eps).astype(
            np.float32
        )
        self.h = np.clip(self.h, self.channel_min, self.channel_max).astype(np.float32)

    def _reset_mec_resource_state(self):
        if self.mec_resource_dynamics == "fixed":
            self.mec_background_load_ratio = 0.0
        else:
            self.mec_background_load_ratio = float(
                np.random.uniform(self.mec_background_load_min, self.mec_background_load_max)
            )
            self.mec_background_load_ratio = float(
                np.clip(
                    self.mec_background_load_ratio,
                    self.mec_background_load_min,
                    self.mec_background_load_max,
                )
            )
        self.mec_resource_ratio = float(1.0 - self.mec_background_load_ratio)
        self.bs_cpu = float(self.bs_cpu_nominal * self.mec_resource_ratio)

    def _advance_mec_resource_state(self):
        if self.mec_resource_dynamics == "fixed":
            self.mec_background_load_ratio = 0.0
        else:
            rho = float(self.mec_resource_corr)
            innovation_scale = self.mec_resource_noise_std * np.sqrt(max(1.0 - rho * rho, 0.0))
            eps_b = float(np.random.normal(0.0, 1.0))
            self.mec_background_load_ratio = (
                self.mec_background_load_base
                + rho * (self.mec_background_load_ratio - self.mec_background_load_base)
                + innovation_scale * eps_b
            )
            self.mec_background_load_ratio = float(
                np.clip(
                    self.mec_background_load_ratio,
                    self.mec_background_load_min,
                    self.mec_background_load_max,
                )
            )
        self.mec_resource_ratio = float(1.0 - self.mec_background_load_ratio)
        self.bs_cpu = float(self.bs_cpu_nominal * self.mec_resource_ratio)

    def _sample_one_shot_tasks(self):
        arrival_low = self.arrival_low_base * self.load_scale
        arrival_high = self.arrival_high_base * self.load_scale
        self.current_task_bits = np.random.uniform(arrival_low, arrival_high, size=self.N).astype(
            np.float32
        )
        cycles_factor = np.random.uniform(
            self.task_cycles_low_factor, self.task_cycles_high_factor, size=self.N
        ).astype(np.float32)
        self.current_task_cycles_per_bit = (self.cycles_per_bit * cycles_factor).astype(np.float32)
        self.current_task_deadline = self.delay_budget_vec.copy().astype(np.float32)
        cycle_span = self.task_cycles_high_factor - self.task_cycles_low_factor
        if cycle_span < 1e-12:
            cycle_state = np.ones(self.N, dtype=np.float32)
        else:
            cycle_state = ((cycles_factor - self.task_cycles_low_factor) / cycle_span).astype(
                np.float32
            )
        urgency_state = 1.0 - np.clip(self.delay_budget_state, 0.0, 1.0)
        self.current_task_profile_state = np.clip(
            0.5 * cycle_state + 0.5 * urgency_state, 0.0, 1.0
        ).astype(np.float32)

    def reset(self):
        if self.initial_queue_ratio > 0:
            if self.initial_queue_profile == "tiered":
                queue_profile = self._make_device_profile(
                    self.initial_queue_heterogeneity_span, shuffle=True
                )
                if self.initial_queue_jitter > 1e-12:
                    jitter_low = max(1.0 - self.initial_queue_jitter, 0.0)
                    jitter_high = 1.0 + self.initial_queue_jitter
                    queue_profile = queue_profile * np.random.uniform(
                        jitter_low, jitter_high, size=self.N
                    ).astype(np.float32)
                    queue_profile = queue_profile / (float(np.mean(queue_profile)) + 1e-12)
                initial_ratios = self.initial_queue_ratio * queue_profile
            else:
                initial_ratios = np.random.uniform(
                    0.5 * self.initial_queue_ratio, 1.5 * self.initial_queue_ratio, size=self.N
                ).astype(np.float32)
            initial_queues = initial_ratios * self.max_buffer
            self.queues = np.clip(initial_queues, 0, self.max_buffer).astype(np.float32)
            self.initial_queue_ratio_vec = np.clip(initial_ratios, 0.0, 1.0).astype(np.float32)
        else:
            self.queues = np.zeros(self.N, dtype=np.float32)
            self.initial_queue_ratio_vec = np.zeros(self.N, dtype=np.float32)
        self.queues = np.zeros(self.N, dtype=np.float32)
        self.initial_queue_ratio_vec = np.zeros(self.N, dtype=np.float32)
        self.last_total_delay = 0.0
        self.last_avg_delay = 0.0
        self.last_delay_norm_ratio = 0.0
        self.last_delay_margin = 0.0
        self.last_queue_margin = 0.0
        self.last_snr_margin = 0.0
        self.last_rsu_load_norm = np.zeros(self.R, dtype=np.float32)
        self.last_service_failure = 0.0
        self.last_energy_norm = 0.0
        self.last_offload_ratio = 0.0
        self.step_count = 0
        self.ep_step_count = 0
        self._reset_channel_state()
        self._reset_mec_resource_state()
        self._sample_one_shot_tasks()
        return self._get_state()

    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-x))

    def _a_to_rsu_index(self, a_ap: np.ndarray) -> np.ndarray:
        return a_ap_to_rsu_index(a_ap, self.R, mode=self.ap_projection_mode)

    def _decode_hybrid_ap(self, ap_flat: np.ndarray) -> np.ndarray:
        ap = np.asarray(ap_flat, dtype=np.float32).reshape(self.N, self.R)
        rsu_idx = np.argmax(ap, axis=1).astype(np.int64)
        row_sum = np.sum(ap, axis=1, keepdims=True)
        if np.all(ap >= -1e-06) and np.all(row_sum > 1e-09):
            prob = np.clip(ap, 0.0, None)
            prob = prob / (np.sum(prob, axis=1, keepdims=True) + 1e-09)
        else:
            logits = ap - np.max(ap, axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            prob = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + 1e-09)
        entropy = -np.sum(prob * np.log(prob + 1e-09), axis=1)
        self.last_ap_entropy = float(np.mean(entropy))
        self.last_ap_max_prob = float(np.mean(np.max(prob, axis=1)))
        return rsu_idx

    def _update_ap_load_diagnostics(self, rsu_idx: np.ndarray, offload_active: np.ndarray):
        rsu_counts = np.bincount(rsu_idx[offload_active], minlength=self.R).astype(np.float32)
        rsu_load_norm = rsu_counts / max(float(self.N), 1.0)
        self.last_rsu_load_norm = np.clip(rsu_load_norm, 0.0, 1.0)
        total_active = float(np.sum(rsu_counts))
        if total_active > 1e-09:
            self.last_ap_load_distribution = (rsu_counts / total_active).astype(np.float32)
        else:
            self.last_ap_load_distribution = np.zeros(self.R, dtype=np.float32)
        return rsu_counts

    def _decode_action_components(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).flatten()
        if action.shape[0] != self.action_dim:
            raise ValueError(f"Expected action dim {self.action_dim}, got {action.shape[0]}")
        a_offload = action[: self.N]
        a_power = action[self.N : 2 * self.N]
        ap_flat = action[2 * self.N : 2 * self.N + self.N * self.R]
        tx_power_vec = self._decode_tx_power(a_power)
        offload_ratio = np.clip((a_offload + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
        if self.action_ablation == "no_rel":
            redundancy = np.zeros(self.N, dtype=np.float32)
        else:
            redundancy = np.zeros(self.N, dtype=np.float32)
        if self.action_ablation == "no_ap":
            rsu_idx = self._least_load_rsu_index(offload_ratio > 1e-06)
            self.last_ap_entropy = 0.0
            self.last_ap_max_prob = 1.0
        else:
            rsu_idx = self._decode_hybrid_ap(ap_flat)
        return (offload_ratio, offload_ratio.copy(), redundancy, rsu_idx, tx_power_vec)

    def _least_load_rsu_index(self, offload_active: np.ndarray) -> np.ndarray:
        idx = np.zeros(self.N, dtype=np.int64)
        if self.R <= 1:
            return idx
        counts = np.zeros(self.R, dtype=np.int64)
        for dev in np.flatnonzero(offload_active):
            chosen = int(np.argmin(counts))
            idx[dev] = chosen
            counts[chosen] += 1
        return idx

    def step(self, action):
        self.step_count += 1
        self.ep_step_count += 1
        return self._step_oneshot(action)

    def _step_oneshot(self, action):
        offload_ratio, raw_offload_ratio, redundancy, rsu_idx, tx_power_vec = (
            self._decode_action_components(action)
        )
        task_bits = self.current_task_bits.astype(np.float32)
        task_cpb = self.current_task_cycles_per_bit.astype(np.float32)
        deadline_vec = np.full(self.N, max(self.slot_duration, 1e-09), dtype=np.float32)
        local_bits = ((1.0 - offload_ratio) * task_bits).astype(np.float32)
        requested_edge = (offload_ratio * task_bits).astype(np.float32)
        offload_active = requested_edge > 1e-06
        rsu_counts = self._update_ap_load_diagnostics(rsu_idx, offload_active)
        snr = tx_power_vec * self.h / self.noise
        rsu_share = np.maximum(rsu_counts[rsu_idx], 1.0)
        bw_i = self.rsu_access_bw[rsu_idx] / rsu_share
        access_rate = bw_i * np.log2(1.0 + snr)
        backhaul_rate = self.backhaul_bw / rsu_share
        local_cycles = (local_bits * task_cpb).astype(np.float32)
        local_capacity_bits = (self.local_cpu * deadline_vec / (task_cpb + 1e-09)).astype(
            np.float32
        )
        local_success = np.minimum(local_bits, local_capacity_bits).astype(np.float32)
        local_failure = np.maximum(local_bits - local_success, 0.0).astype(np.float32)
        edge_arrival_window = np.clip(
            deadline_vec * self.ul_time_fraction, 0.0, deadline_vec
        ).astype(np.float32)
        mec_service_window = np.maximum(deadline_vec - edge_arrival_window, 0.0).astype(np.float32)
        fixed_edge_delay = np.where(offload_active, self.backhaul_delay, 0.0).astype(np.float32)
        arrival_per_bit_time = 1.0 / (access_rate + 1e-09) + 1.0 / (backhaul_rate + 1e-09)
        edge_arrival_capacity = np.where(
            offload_active,
            np.maximum(edge_arrival_window - fixed_edge_delay, 0.0)
            / (arrival_per_bit_time + 1e-12),
            0.0,
        ).astype(np.float32)
        edge_arrived = np.minimum(requested_edge, edge_arrival_capacity).astype(np.float32)
        edge_arrival_failure = np.maximum(requested_edge - edge_arrived, 0.0).astype(np.float32)
        arrived_cycles = (edge_arrived * task_cpb).astype(np.float32)
        mec_cycle_budget = float(self.bs_cpu * np.mean(mec_service_window))
        sum_arrived_cycles = float(np.sum(arrived_cycles))
        if sum_arrived_cycles < 1e-09:
            admitted_cycles = np.zeros(self.N, dtype=np.float32)
        elif sum_arrived_cycles <= mec_cycle_budget:
            admitted_cycles = arrived_cycles.copy()
        else:
            admitted_cycles = (
                arrived_cycles / (sum_arrived_cycles + 1e-09) * mec_cycle_budget
            ).astype(np.float32)
            admitted_cycles = np.minimum(admitted_cycles, arrived_cycles).astype(np.float32)
        edge_admitted = np.minimum(edge_arrived, admitted_cycles / (task_cpb + 1e-09)).astype(
            np.float32
        )
        edge_admission_failure = np.maximum(edge_arrived - edge_admitted, 0.0).astype(np.float32)
        edge_success = edge_admitted.copy()
        edge_deadline_failure = np.zeros(self.N, dtype=np.float32)
        offloading_delivery_shortfall = (
            edge_arrival_failure + edge_admission_failure + edge_deadline_failure
        ).astype(np.float32)
        total_failed_bits_vec = (local_failure + offloading_delivery_shortfall).astype(np.float32)
        total_success_bits_vec = (local_success + edge_success).astype(np.float32)
        sum_requested = float(np.sum(requested_edge))
        sum_task_bits = float(np.sum(task_bits))
        delivery_failure_ratio = float(
            np.sum(offloading_delivery_shortfall) / (sum_requested + 1e-09)
        )
        service_failure_ratio = float(np.sum(total_failed_bits_vec) / (sum_task_bits + 1e-09))
        completed_workload_ratio = float(np.sum(total_success_bits_vec) / (sum_task_bits + 1e-09))
        outage_ratio = float(np.clip(service_failure_ratio, 0.0, 1.0))
        cost_failed_bits_vec = total_failed_bits_vec
        oneshot_cost_ratio = float(np.sum(cost_failed_bits_vec) / (sum_task_bits + 1e-09))
        edge_cycles = (requested_edge * task_cpb).astype(np.float32)
        local_delay = local_cycles / (self.local_cpu + 1e-09)
        tx_time = requested_edge / (access_rate + 1e-09)
        backhaul_time = np.where(
            requested_edge > 1e-09,
            requested_edge / (backhaul_rate + 1e-09) + self.backhaul_delay,
            0.0,
        )
        requested_cycles = (requested_edge * task_cpb).astype(np.float32)
        sum_requested_cycles = float(np.sum(requested_cycles))
        if sum_requested_cycles > 1e-09:
            requested_mec_rate = (
                self.bs_cpu * requested_cycles / (sum_requested_cycles + 1e-09)
            ).astype(np.float32)
            mec_compute = np.where(
                requested_edge > 1e-09, edge_cycles / (requested_mec_rate + 1e-09), 0.0
            )
        else:
            mec_compute = np.zeros(self.N, dtype=np.float32)
        offload_delay = np.where(offload_active, tx_time + backhaul_time + mec_compute, 0.0)
        delay_i = np.maximum(local_delay, offload_delay)
        total_delay = float(np.mean(delay_i))
        delay_ratio_i = delay_i / deadline_vec
        delay_fail_ratio = service_failure_ratio
        delay_over_i = np.clip(delay_ratio_i - 1.0, 0.0, 1.0)
        delay_over = float(np.mean(delay_over_i))
        delay_norm_ratio = float(np.mean(delay_ratio_i))
        queue_violation = 0.0
        queue_ratio = 0.0
        soft_bomb = 0.0
        impl_cost = float(np.clip(oneshot_cost_ratio, 0.0, 1.0))
        paper_cost = impl_cost
        cost = impl_cost
        local_energy = self._compute_local_energy(local_success, task_cpb)
        energy_bits = (
            edge_arrived
            if getattr(self, "tx_energy_accounting", "attempted") == "delivered"
            else requested_edge
        )
        tx_energy = self._compute_tx_energy(energy_bits, access_rate, tx_power_vec=tx_power_vec)
        queue_holding_energy = 0.0
        total_energy = local_energy + tx_energy
        avg_energy = total_energy / max(float(self.N), 1.0)
        reward_energy_penalty = float(avg_energy / self.energy_ref)
        if self.reward_delay_mode == "normalized":
            reward_delay_penalty = float(self.delay_budget * delay_norm_ratio)
        else:
            reward_delay_penalty = float(total_delay)
        reward = -reward_energy_penalty if self.reward_mode == "energy" else -reward_delay_penalty
        self.queues = np.zeros(self.N, dtype=np.float32)
        self.last_total_delay = total_delay
        self.last_avg_delay = float(np.mean(delay_i))
        self.last_delay_norm_ratio = float(delay_norm_ratio)
        self.last_delay_margin = float(np.clip(np.mean(1.0 - delay_ratio_i), -1.0, 1.0))
        self.last_queue_margin = 1.0
        avg_snr = float(np.mean(snr))
        self.last_snr_margin = float(
            np.clip(avg_snr / (self.snr_threshold_linear + 1e-09), -1.0, 1.0)
        )
        self.last_service_failure = float(np.clip(service_failure_ratio, 0.0, 1.0))
        self.last_energy_norm = float(np.clip(reward_energy_penalty, 0.0, 1.0))
        self.last_offload_ratio = float(np.clip(np.mean(offload_ratio), 0.0, 1.0))
        mec_resource_ratio_used = float(self.mec_resource_ratio)
        effective_bs_cpu_used = float(self.bs_cpu)
        self._advance_channel_state()
        self._advance_mec_resource_state()
        self._sample_one_shot_tasks()
        next_state = self._get_state()
        done = self.ep_step_count >= self.max_ep_steps
        info = {
            "N": self.N,
            "task_model": self.task_model,
            "reward": float(reward),
            "reward_mode": self.reward_mode,
            "reward_delay_penalty": float(reward_delay_penalty),
            "reward_delay_mode": self.reward_delay_mode,
            "reward_energy_penalty": float(reward_energy_penalty),
            "energy": float(avg_energy),
            "total_energy": float(total_energy),
            "local_energy": float(local_energy),
            "tx_energy": float(tx_energy),
            "local_energy_model": self.local_energy_model,
            "local_energy_zeta_bar": float(self.local_energy_zeta_bar),
            "local_cycles_per_bit_ref": float(self.local_cycles_per_bit_ref),
            "tx_energy_model": self.tx_energy_model,
            "tx_energy_rate_sensitivity": float(self.tx_energy_rate_sensitivity),
            "tx_energy_rate_min_multiplier": float(self.tx_energy_rate_min_multiplier),
            "tx_energy_rate_max_multiplier": float(self.tx_energy_rate_max_multiplier),
            "tx_energy_ref_rate_mode": self.tx_energy_ref_rate_mode,
            "tx_energy_ref_rate": float(self.last_tx_energy_ref_rate),
            "tx_energy_time_scale": float(self.tx_energy_time_scale),
            "tx_pa_efficiency": float(self.tx_pa_efficiency),
            "tx_circuit_power": float(self.tx_circuit_power),
            "tx_power_control": self.tx_power_control,
            "tx_power_base": float(self.tx_power_base),
            "tx_power_mean": float(np.mean(tx_power_vec)),
            "avg_tx_power": float(np.mean(tx_power_vec)),
            "tx_power_min_used": float(np.min(tx_power_vec)),
            "tx_power_max_used": float(np.max(tx_power_vec)),
            "avg_tx_energy": float(tx_energy),
            "avg_local_energy": float(local_energy),
            "avg_energy": float(avg_energy),
            "E_tx_over_E_loc": float(tx_energy / (local_energy + 1e-12)),
            "E_loc_share": float(local_energy / (total_energy + 1e-12)),
            "E_tx_share": float(tx_energy / (total_energy + 1e-12)),
            "queue_holding_energy": float(queue_holding_energy),
            "local_energy_per_bit": float(self.local_energy_per_bit),
            "tx_energy_per_bit": float(self.tx_energy_per_bit),
            "queue_holding_energy_per_bit": float(self.queue_holding_energy_per_bit),
            "energy_heterogeneity_span": float(self.energy_heterogeneity_span),
            "energy_factor_mean": float(np.mean(self.energy_factor)),
            "energy_factor_min": float(np.min(self.energy_factor)),
            "energy_factor_max": float(np.max(self.energy_factor)),
            "energy_ref": float(self.energy_ref),
            "cost": float(cost),
            "impl_cost": float(impl_cost),
            "paper_cost": float(paper_cost),
            "total_delay": float(total_delay),
            "avg_delay_per_device": float(np.mean(delay_i)),
            "raw_delay": float(np.mean(delay_i)),
            "delay_norm": float(delay_norm_ratio),
            "delay_budget_mean": float(np.mean(deadline_vec)),
            "delay_budget_min": float(np.min(deadline_vec)),
            "delay_budget_max": float(np.max(deadline_vec)),
            "outage_ratio": float(outage_ratio),
            "transmission_shortfall_ratio": float(
                np.sum(edge_arrival_failure) / (sum_requested + 1e-09)
            ),
            "transmission_shortfall_bits": float(np.sum(edge_arrival_failure)),
            "access_shortfall_ratio": float(np.sum(edge_arrival_failure) / (sum_requested + 1e-09)),
            "access_shortfall_bits": float(np.sum(edge_arrival_failure)),
            "edge_admission_shortfall_ratio": float(
                np.sum(edge_admission_failure) / (sum_requested + 1e-09)
            ),
            "edge_admission_shortfall_bits": float(np.sum(edge_admission_failure)),
            "offloading_delivery_failure_ratio": float(delivery_failure_ratio),
            "offloading_delivery_failure_bits": float(np.sum(offloading_delivery_shortfall)),
            "service_failure_ratio": float(service_failure_ratio),
            "service_failure_bits": float(np.sum(total_failed_bits_vec)),
            "completed_workload_ratio": float(completed_workload_ratio),
            "completed_workload_bits": float(np.sum(total_success_bits_vec)),
            "local_failure_ratio": float(np.sum(local_failure) / (sum_task_bits + 1e-09)),
            "local_failure_bits": float(np.sum(local_failure)),
            "edge_arrival_failure_ratio": float(
                np.sum(edge_arrival_failure) / (sum_task_bits + 1e-09)
            ),
            "edge_arrival_failure_bits": float(np.sum(edge_arrival_failure)),
            "mec_admission_failure_ratio": float(
                np.sum(edge_admission_failure) / (sum_task_bits + 1e-09)
            ),
            "mec_admission_failure_bits": float(np.sum(edge_admission_failure)),
            "mec_deadline_failure_ratio": float(
                np.sum(edge_deadline_failure) / (sum_task_bits + 1e-09)
            ),
            "mec_deadline_failure_bits": float(np.sum(edge_deadline_failure)),
            "requested_edge_bits": float(sum_requested),
            "uploaded_edge_bits": float(np.sum(edge_arrived)),
            "edge_success_bits": float(np.sum(edge_success)),
            "local_success_bits": float(np.sum(local_success)),
            "delay_fail_ratio": float(delay_fail_ratio),
            "delay_over": float(delay_over),
            "queue_violation": float(queue_violation),
            "queue_ratio": float(queue_ratio),
            "raw_offload_ratio_mean": float(np.mean(raw_offload_ratio)),
            "executed_offload_ratio_mean": float(np.mean(offload_ratio)),
            "projected_edge_bits": 0.0,
            "shortfall_projection_margin": 0.0,
            "shortfall_projection_scope": self.shortfall_projection_scope,
            "redundancy_mean": float(np.mean(redundancy)),
            "soft_bomb": float(soft_bomb),
            "oneshot_cost_components": self.oneshot_cost_components,
            "risk_cost_mode": self.risk_cost_mode,
            "action_mode": self.action_mode,
            "hybrid_ap_enabled": bool(self.hybrid_ap_enabled),
            "ap_entropy": float(self.last_ap_entropy),
            "avg_ap_entropy": float(self.last_ap_entropy),
            "ap_confidence": float(self.last_ap_max_prob),
            "avg_ap_max_prob": float(self.last_ap_max_prob),
            "ap_load_distribution": self.last_ap_load_distribution.copy(),
            "channel_dynamics": self.channel_dynamics,
            "mec_resource_dynamics": self.mec_resource_dynamics,
            "mec_resource_ratio": float(mec_resource_ratio_used),
            "mec_background_load_ratio": float(1.0 - mec_resource_ratio_used),
            "mec_resource_ratio_min_eff": float(self.mec_resource_ratio_min_eff),
            "mec_resource_ratio_max_eff": float(self.mec_resource_ratio_max_eff),
            "effective_bs_cpu": float(effective_bs_cpu_used),
            "bs_cpu_nominal": float(self.bs_cpu_nominal),
            "edge_arrival_window_mean": float(np.mean(edge_arrival_window)),
            "mec_service_window_mean": float(np.mean(mec_service_window)),
            "next_mec_resource_ratio": float(self.mec_resource_ratio),
            "last_service_failure_feedback": float(self.last_service_failure),
            "last_energy_feedback": float(self.last_energy_norm),
            "last_offload_feedback": float(self.last_offload_ratio),
            "rsu_load_norm": self.last_rsu_load_norm.copy(),
            "delay_budget_per_device": deadline_vec.copy(),
            "device_energy_factor": self.energy_factor.copy(),
            "local_energy_per_bit_vec": self.local_energy_per_bit_vec.copy(),
            "tx_energy_per_bit_vec": self.tx_energy_per_bit_vec.copy(),
            "queue_holding_energy_per_bit_vec": self.queue_holding_energy_per_bit_vec.copy(),
            "initial_queue_ratio_vec": self.initial_queue_ratio_vec.copy(),
            "task_bits_per_device": task_bits.copy(),
            "task_cycles_per_bit": task_cpb.copy(),
        }
        return (next_state, reward, False if done is False else True, info)

    def _get_state(self):
        h_norm = (self.h - 0.5) / 0.5
        h_norm = np.clip(h_norm, 0.0, 1.0)
        task_ref = max(self.arrival_high_base * self.load_scale, 1e-09)
        q_norm = self.current_task_bits / task_ref
        q_norm = np.clip(q_norm, 0.0, 1.0)
        device_profile_norm = np.clip(self.current_task_profile_state, 0.0, 1.0).astype(np.float32)
        rsu_load = np.clip(self.last_rsu_load_norm, 0.0, 1.0).astype(np.float32)
        resource_span = self.mec_resource_ratio_max_eff - self.mec_resource_ratio_min_eff
        if resource_span < 1e-12:
            mec_resource = np.array([1.0], dtype=np.float32)
        else:
            mec_resource = np.array(
                [
                    np.clip(
                        (self.mec_resource_ratio - self.mec_resource_ratio_min_eff)
                        / (resource_span + 1e-12),
                        0.0,
                        1.0,
                    )
                ],
                dtype=np.float32,
            )
        prev_feedback = np.array(
            [
                np.clip(self.last_service_failure, 0.0, 1.0),
                np.clip(self.last_energy_norm, 0.0, 1.0),
                np.clip(self.last_offload_ratio, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        delay_norm = np.array([min(self.last_delay_norm_ratio, 1.0)], dtype=np.float32)
        delay_margin = np.array([self.last_delay_margin], dtype=np.float32)
        queue_margin = np.array([self.last_queue_margin], dtype=np.float32)
        snr_margin = np.array([self.last_snr_margin], dtype=np.float32)
        state = np.concatenate(
            [
                h_norm,
                q_norm,
                device_profile_norm,
                rsu_load,
                mec_resource,
                prev_feedback,
                delay_norm,
                delay_margin,
                queue_margin,
                snr_margin,
            ]
        ).astype(np.float32)
        return state
