"""Training arms for the IoT-J revision study (CPO, E1-E3, E6, fair critics, model mismatch).

Every arm starts from the frozen v2 configuration (`cloud24.plan.configure`) and changes only what
its name says. Names:

    sfr                         SFR-SAC as selected (split service-flow actor, structured critics, hinge)
    sfr_hinge_d0.08 / _d0.12    SFR-SAC at another budget d
    sfr_linear_d0.08/0.10/0.12  SFR-SAC with the untruncated Lagrangian actor penalty (E2)
    sfr_maxgain                 SFR-SAC whose AP is the best-channel AP (x and p still learned) (E3)
    sfr_apind                   SFR-SAC with AP-independent channel gains (E3)
    td3, td3_apind              Safe TD3 as in the main grid (MLP actor and critics)
    td3_struct                  Safe TD3 with SFR-SAC's structured critics (MLP actor)
    td3_struct_split            Safe TD3 with SFR-SAC's structured critics and split actor network
    pen_struct_b<beta>          fixed penalty beta with SFR-SAC's critics and actor (no dual update)
    sacmlp_g0 / sacmlp_g0.99    Safe SAC with the split actor and MLP critics, gamma 0 or 0.99 (E1)
    td3_g0 / td3_g0.99          Safe TD3 (MLP), gamma 0 or 0.99 (E1)
    pen5_g0 / pen5_g0.99        Penalty SAC beta=5 (shared service-flow actor, MLP critic) (E1)
    cpo_g0 / cpo_g0.99          CPO with the split service-flow actor, gamma 0 or 0.99 (E7)
    mb_greedy, mb_greedy_apind  model-based per-slot greedy (full simulator information)
    mm_<case>_<arm>             model mismatch: the true environment differs from the constants
                                given to SFR-SAC's structured networks (arms: sfr_wrong,
                                sfr_right, td3, mb)
"""
from __future__ import annotations

import copy

from cloud24 import plan

PAIRS5 = tuple((1000 + i, i) for i in range(5))
PAIRS3 = PAIRS5[:3]

CPO = dict(gamma=None, gae_lambda=0.95, cost_gamma=None, target_kl=0.01, damping=0.1, cg_iters=10,
           backtrack_iters=10, backtrack_coeff=0.8, vf_lr=0.001, vf_iters=80, vf_batch=256,
           steps_per_update=4000, fisher_fraction=0.2)

# True environment change, and the nominal value SFR-SAC's structured networks keep assuming.
MISMATCH = {
    "mec_capacity": dict(environment=dict(bs_cpu_base=0.8 * 8.0e10), assumptions=dict(bs_cpu_base=8.0e10)),
    "pa_efficiency": dict(environment=dict(tx_pa_efficiency=0.25), assumptions=dict(tx_pa_efficiency=0.35)),
    "cycles": dict(environment=dict(cycles_per_bit=3600.0), assumptions=dict(cycles_per_bit=3000.0)),
    # Structural: the MEC shares capacity max-min fairly; the critics assume proportional sharing.
    "mec_sharing": dict(environment=dict(mec_sharing="max_min_fair"), assumptions={}),
}
MISMATCH_ARMS = ("sfr_wrong", "sfr_right", "td3", "mb")
PENALTY_BETAS = (2.0, 2.5, 3.0, 3.5, 4.5)
CPU_METHODS = {"cpo", "lyapunov", "const_ratio", "local_first", "mb_greedy"}


def _base(variant, n, layout, seed, small):
    return plan.configure(variant, n=n, layout=layout, train_seed=seed, selected="split_exact", small=small)


def _cpu(cfg):
    cfg["execution"] = dict(device="cpu", require_cuda=False, tf32=False)


def _discounted(cfg):
    cfg["training"].update(gamma=0.99, reward_scale="auto")


def _legacy_entropy(cfg, method):
    # The exact AP-entropy controls require gamma = 0; E1 uses the sampled form in both arms.
    cfg["algorithms"][method]["ap_entropy_mode"] = "legacy_sampled"


def _structured_td3(cfg, split_actor):
    safe, td3 = cfg["algorithms"]["safe"], cfg["algorithms"]["td3_lag"]
    td3["critic_architecture"] = "service_flow"
    for key in ("energy_hidden", "cost_delivery_hidden", "energy_component_loss_weight", "cost_delivery_loss_weight"):
        td3[key] = safe[key]
    if split_actor:
        # The td3 base keeps the shared actor setting; SFR-SAC uses the split service-flow network.
        td3["actor_architecture"] = "split_service_flow"
        for key in ("actor_hidden", "actor_embedding_dim"):
            td3[key] = safe[key]


def configure(variant, *, n=70, layout=1000, seed=0, small=False):
    """Return the full configuration of one training arm."""
    if variant == "sfr":
        cfg = _base("split_exact", n, layout, seed, small)
    elif variant.startswith("sfr_hinge_d") or variant.startswith("sfr_linear_d"):
        cfg = _base("split_exact", n, layout, seed, small)
        mode, d = variant[4:].split("_d")
        cfg["cost_limit"] = float(d)
        cfg["algorithms"]["safe"]["penalty_mode"] = mode
    elif variant == "sfr_maxgain":
        cfg = _base("split_exact", n, layout, seed, small)
        cfg["algorithms"]["safe"]["actor_ap_rule"] = "max_gain"
    elif variant == "sfr_apind":
        cfg = _base("split_exact", n, layout, seed, small)
        cfg["environment"]["ap_dependent_gain"] = False
    elif variant in {"td3", "td3_g0"}:
        cfg = _base("td3", n, layout, seed, small)
    elif variant == "td3_apind":
        cfg = _base("td3", n, layout, seed, small)
        cfg["environment"]["ap_dependent_gain"] = False
    elif variant in {"td3_struct", "td3_struct_split"}:
        cfg = _base("td3", n, layout, seed, small)
        _structured_td3(cfg, split_actor=variant.endswith("split"))
    elif variant.startswith("pen_struct_b"):
        beta = float(variant[len("pen_struct_b"):])
        cfg = _base("split_exact", n, layout, seed, small)
        safe = cfg["algorithms"]["safe"]
        # gamma = 0: -Q_r + (beta/cost_coef)*cost_coef*(Q_c - d) is the fixed-penalty objective r - beta*c.
        safe.update(penalty_mode="linear", lambda_mode="fixed", fixed_lambda=beta / safe["cost_coef"],
                    initial_lambda=beta / safe["cost_coef"])
    elif variant in {"sacmlp_g0", "sacmlp_g0.99"}:
        cfg = _base("mlp_critics", n, layout, seed, small)
        _legacy_entropy(cfg, "safe")
        if variant.endswith("0.99"):
            _discounted(cfg)
    elif variant == "td3_g0.99":
        cfg = _base("td3", n, layout, seed, small)
        _discounted(cfg)
    elif variant in {"pen5_g0", "pen5_g0.99"}:
        cfg = plan.configure("penalty_beta5", n=n, layout=layout, train_seed=seed, selected="shared_exact",
                             small=small)
        _legacy_entropy(cfg, "penalty_sac")
        if variant.endswith("0.99"):
            _discounted(cfg)
    elif variant in {"cpo_g0", "cpo_g0.99"}:
        cfg = _base("split_exact", n, layout, seed, small)
        safe = cfg["algorithms"]["safe"]
        cpo = dict(CPO)
        for key in ("actor_architecture", "actor_hidden", "actor_embedding_dim", "ap_residual_bound"):
            if key in safe:
                cpo[key] = safe[key]
        cfg["algorithms"]["cpo"] = cpo
        cfg["method"] = "cpo"
        if small:
            cpo.update(steps_per_update=24, vf_iters=3, cg_iters=3)
        if variant.endswith("0.99"):
            _discounted(cfg)
    elif variant in {"mb_greedy", "mb_greedy_apind"}:
        cfg = _base("mb_greedy", n, layout, seed, small)
        if variant.endswith("apind"):
            cfg["environment"]["ap_dependent_gain"] = False
    elif variant.startswith("mm_"):
        case, arm = None, None
        for name in MISMATCH:
            if variant.startswith(f"mm_{name}_"):
                case, arm = name, variant[len(f"mm_{name}_"):]
        if case is None or arm not in MISMATCH_ARMS:
            raise ValueError(f"Unknown mismatch arm: {variant}")
        spec = MISMATCH[case]
        if arm == "sfr_right" and not spec["assumptions"]:
            raise ValueError("The structural mismatch has no correctly specified SFR-SAC arm")
        base = {"sfr_wrong": "split_exact", "sfr_right": "split_exact", "td3": "td3", "mb": "mb_greedy"}[arm]
        cfg = _base(base, n, layout, seed, small)
        cfg["environment"].update(spec["environment"])
        if arm == "sfr_wrong" and spec["assumptions"]:
            cfg["algorithms"]["safe"]["model_assumptions"] = dict(spec["assumptions"])
    else:
        raise ValueError(f"Unknown revision variant: {variant}")
    if cfg["method"] in CPU_METHODS or small:
        _cpu(cfg)
    cfg["revision_variant"] = variant
    return copy.deepcopy(cfg)
