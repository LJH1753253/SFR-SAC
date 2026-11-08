"""Independent continuous and categorical encoders for the cloud study."""

import copy

import torch

from agents.structured_actor import ServiceFlowActor


class SplitServiceFlowActor(ServiceFlowActor):
    def __init__(self, *args, ap_residual_bound=None, ap_rule="learned", **kwargs):
        super().__init__(*args, **kwargs)
        if ap_rule not in {"learned", "max_gain"}:
            raise ValueError("ap_rule must be 'learned' or 'max_gain'")
        # "max_gain" executes the best-channel AP (revision E3); x and p are still learned.
        self.ap_rule = ap_rule
        self.ap_device_encoder = copy.deepcopy(self.device_encoder)
        self.ap_global_encoder = copy.deepcopy(self.global_encoder)
        self.ap_device_embedding = copy.deepcopy(self.device_embedding)
        self.ap_residual_bound = ap_residual_bound
        if ap_residual_bound is not None and ap_residual_bound <= 0:
            raise ValueError("ap_residual_bound must be positive")

    def _context(self, features, links, load, scalars, resource, categorical):
        encoder = self.ap_device_encoder if categorical else self.device_encoder
        global_encoder = self.ap_global_encoder if categorical else self.global_encoder
        embedding = self.ap_device_embedding if categorical else self.device_embedding
        identity = embedding.weight.unsqueeze(0).expand(features.shape[0], -1, -1)
        if not self.use_device_embedding:
            identity = torch.zeros_like(identity)
        tokens = encoder(torch.cat((features, links, identity), -1))
        pooled = torch.cat((tokens.mean(1), tokens.max(1).values, load, scalars, resource), 1)
        global_token = global_encoder(pooled).unsqueeze(1).expand(-1, self.num_devices, -1)
        if not self.use_pooled_context:
            global_token = torch.zeros_like(global_token)
        return torch.cat((tokens, global_token), -1)

    def forward(self, state):
        n, m = self.num_devices, self.num_rsu
        features, links, load, scalars, resource = self.observed_features(state)
        if not self.use_physics_features:
            features = features.clone()
            features[:, :, 3:5] = 0
            resource = torch.zeros_like(resource)
        continuous_context = self._context(features, links, load, scalars, resource, False)
        ap_context = self._context(features, links, load, scalars, resource, True)
        continuous = self.continuous_head(continuous_context)
        mean = torch.cat((continuous[:, :, 0], continuous[:, :, 1]), 1)
        log_std = torch.cat((continuous[:, :, 2], continuous[:, :, 3]), 1).clamp(
            self.LOG_STD_MIN, self.LOG_STD_MAX)
        pair = torch.stack((links, links-links.max(-1, keepdim=True).values,
                            load.unsqueeze(1).expand(-1, n, -1),
                            links.mean(1, keepdim=True).expand(-1, n, -1)), -1)
        residual = self.ap_head(torch.cat((ap_context.unsqueeze(2).expand(-1, -1, m, -1), pair), -1)).squeeze(-1)
        if self.ap_residual_bound is not None:
            bound = float(self.ap_residual_bound)
            residual = bound * torch.tanh(residual / bound)
        logits = residual if self.ap_prior is None else residual + self.ap_prior(state)
        if self.ap_rule == "max_gain":
            logits = 50.0 * (links - links.max(-1, keepdim=True).values).detach()
        return mean, log_std, logits


def categorical_value_loss(agent, state, action):
    """One-device counterfactuals at one-hot vertices; no simulator queries."""
    n, m = agent.num_devices, agent.num_rsu
    k = min(n, agent.ap_value_devices)
    indices = (torch.arange(k, device=state.device) + agent.update_step*k) % n
    _, _, logits = agent.actor(state)
    probs = logits[:, indices].softmax(-1)
    batch = state.shape[0]
    values = []
    # Enumerate one device at a time to bound memory independently of N*K*M.
    with torch.no_grad():
        for device_index in indices.tolist():
            alternatives = action.detach().unsqueeze(1).expand(-1, m, -1).clone()
            block = alternatives[:, :, 2*n:].reshape(batch, m, n, m)
            block[:, :, device_index, :] = torch.eye(m, device=state.device)[None]
            flat_action = alternatives.reshape(batch*m, -1)
            flat_state = state.unsqueeze(1).expand(-1, m, -1).reshape(batch*m, -1)
            scores = []
            for start in range(0, batch*m, agent.ap_value_chunk):
                s, a = flat_state[start:start+agent.ap_value_chunk], flat_action[start:start+agent.ap_value_chunk]
                q = torch.minimum(agent.critic_1(s, a), agent.critic_2(s, a))
                c = torch.maximum(agent.cost_critic_1(s, a), agent.cost_critic_2(s, a)).clamp_min(0)
                violation = c-agent.cost_limit
                if agent.penalty_mode == "hinge":
                    violation = violation.clamp_min(0)
                scores.append((-q + agent.lam.detach()*agent.cost_coef*violation).squeeze(-1))
            values.append(torch.cat(scores).reshape(batch, m))
        scores = torch.stack(values, 1)
        scores = scores - scores.mean(-1, keepdim=True)
    # N/K subset scaling estimates the sum of per-device policy derivatives.
    return n*(probs*scores).sum(-1).mean()
