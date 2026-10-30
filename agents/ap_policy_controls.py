"""Categorical entropy controls and a frozen AP-policy diagnostic."""

import copy

import torch
import torch.nn.functional as F
from torch import nn

from agents.networks import HybridActor


ENTROPY_MODES = {"legacy_sampled", "exact_legacy_scale", "exact_branch_mean"}


def validate_entropy_mode(mode, gamma):
    if mode not in ENTROPY_MODES:
        raise ValueError("Unknown ap_entropy_mode")
    if mode != "legacy_sampled" and gamma != 0.:
        raise ValueError("Exact AP actor-entropy controls currently require gamma=0")


def negative_categorical_entropy(logits):
    log_probs = F.log_softmax(logits, dim=-1)
    return (log_probs.exp()*log_probs).sum(-1).mean(-1, keepdim=True)


def sample_actor_entropy(actor, state, mode, continuous_dim):
    if mode == "legacy_sampled":
        action, log_prob = actor.sample(state)
        return action, log_prob, log_prob/continuous_dim
    action, log_prob, info = actor.sample(state, return_info=True)
    categorical = actor.ap_entropy_weight*negative_categorical_entropy(info["ap_logits"])
    continuous = info["log_prob_cont"]/continuous_dim
    entropy = continuous + (categorical/continuous_dim if mode == "exact_legacy_scale" else categorical)
    return action, log_prob, entropy


class FrozenAPPolicy(HybridActor):
    """Keep a complete AP mapping fixed while the continuous policy learns."""

    def __init__(self, policy):
        nn.Module.__init__(self)
        self.policy = policy
        self.ap_policy = copy.deepcopy(policy).requires_grad_(False).eval()
        for name in ("num_devices", "num_rsu", "continuous_dim", "ap_dim", "action_dim",
                     "max_action", "gumbel_tau", "gumbel_tau_min", "ap_entropy_weight",
                     "LOG_STD_MIN", "LOG_STD_MAX"):
            setattr(self, name, getattr(policy, name))
        self.last_sample_info = {}

    def train(self, mode=True):
        super().train(mode)
        self.ap_policy.eval()
        return self

    def forward(self, state):
        mean, log_std, _ = self.policy(state)
        with torch.no_grad():
            _, _, ap_logits = self.ap_policy(state)
        return mean, log_std, ap_logits
