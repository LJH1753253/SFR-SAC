"""Constrained Policy Optimization (Achiam et al., ICML 2017) for the hybrid action space.

Policy: the same HybridActor network as Safe SAC (Gaussian continuous branch, tanh-squashed,
plus one categorical AP distribution per ED). The likelihood used for the policy gradient is
the exact joint likelihood of the pre-tanh Gaussian sample and the sampled AP indices (the tanh
Jacobian cancels in likelihood ratios). The Fisher-vector product uses the KL divergence of the
Gaussian and categorical parts. The constraint is the per-slot average service-failure cost
(J_C <= d), linearized with discounted cost advantages (performance-difference lemma).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.networks import HybridActor

EPS = 1e-8


class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, s):
        return self.net(s).squeeze(-1)


def flat_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model, flat):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[i : i + n].view_as(p))
        i += n


def flat_grad(y, model, retain_graph=False, create_graph=False):
    grads = torch.autograd.grad(y, list(model.parameters()), retain_graph=retain_graph or create_graph,
                                create_graph=create_graph, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                      for g, p in zip(grads, model.parameters())])


def conjugate_gradient(Avp, b, iters=10, tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rr = torch.dot(r, r)
    for _ in range(iters):
        Ap = Avp(p)
        alpha = rr / (torch.dot(p, Ap) + EPS)
        x += alpha * p
        r -= alpha * Ap
        rr_new = torch.dot(r, r)
        if rr_new < tol:
            break
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x


def solve_cpo_step(g, b, inverse_g, inverse_b, gap, target_kl, tol=1e-10):
    """Maximize g^T x with b^T x + gap <= 0 and x^T H x / 2 <= target_kl."""
    radius_sq = 2.0 * float(target_kl)
    if radius_sq <= 0:
        raise ValueError("target_kl must be positive")
    q = max(float(torch.dot(g, inverse_g)), 0.0)
    r = float(torch.dot(b, inverse_g))
    s = max(float(torch.dot(b, inverse_b)), 0.0)
    zero = torch.zeros_like(g)
    unconstrained = np.sqrt(radius_sq / q) * inverse_g if q > tol else zero
    if float(torch.dot(b, unconstrained)) + gap <= tol:
        return unconstrained, {"case": 4 if s <= tol else 3, "nu": 0.0}
    if s <= tol:
        return zero, {"case": 0, "nu": 0.0}
    recovery_scale = np.sqrt(radius_sq / s)
    if gap > np.sqrt(radius_sq * s) + tol:
        return -recovery_scale * inverse_b, {"case": 0, "nu": recovery_scale}
    # Intersect the trust-region ellipsoid with the active constraint plane.
    perpendicular = inverse_g - (r / s) * inverse_b
    perpendicular_norm = max(q - r * r / s, 0.0)
    remaining_radius = max(radius_sq - gap * gap / s, 0.0)
    step = -(gap / s) * inverse_b
    multiplier = 0.0
    if perpendicular_norm > tol and remaining_radius > tol:
        scale = np.sqrt(remaining_radius / perpendicular_norm)
        step = step + scale * perpendicular
        multiplier = 1.0 / scale
    nu = max((r + multiplier * gap) / s, 0.0)
    return step, {"case": 1 if gap > 0 else 2, "nu": nu}


class CPOAgent:
    def __init__(self, state_dim, num_devices, num_rsu, cost_limit=0.10, gamma=0.99, lam=0.95,
                 cost_gamma=0.99, cost_lam=0.95, target_kl=0.01, damping=0.1, cg_iters=10,
                 backtrack_iters=10, backtrack_coeff=0.8, vf_lr=1e-3, vf_iters=80, vf_batch=256,
                 steps_per_update=4000, device="cpu", ap_prior=None):
        self.N, self.M = int(num_devices), int(num_rsu)
        self.device = torch.device(device)
        self.actor = HybridActor(state_dim, self.N, self.M, ap_prior=ap_prior).to(self.device)
        self.vf = ValueNet(state_dim).to(self.device)
        self.vc = ValueNet(state_dim).to(self.device)
        self.vf_opt = torch.optim.Adam(self.vf.parameters(), lr=vf_lr)
        self.vc_opt = torch.optim.Adam(self.vc.parameters(), lr=vf_lr)
        self.cost_limit = float(cost_limit)
        self.gamma, self.lam, self.cgamma, self.clam = gamma, lam, cost_gamma, cost_lam
        self.target_kl, self.damping, self.cg_iters = target_kl, damping, cg_iters
        self.backtrack_iters, self.backtrack_coeff = backtrack_iters, backtrack_coeff
        self.vf_iters, self.vf_batch = vf_iters, vf_batch
        self.steps_per_update = int(steps_per_update)
        self.lambda_val = 0.0  # reported as the constraint multiplier nu*
        self.alpha = 0.0
        self.buf = []

    # ------------------------------------------------------------ acting
    def _dist(self, s):
        mean, log_std, logits = self.actor(s)
        return mean, log_std.exp(), logits

    def sample(self, state):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, std, logits = self._dist(s)
            z = mean + std * torch.randn_like(std)
            u = torch.distributions.Categorical(logits=logits).sample()  # (1, N)
        onehot = F.one_hot(u, self.M).float().view(1, -1)
        action = torch.cat([torch.tanh(z), onehot], dim=1).squeeze(0).cpu().numpy()
        return action, z.squeeze(0).cpu().numpy(), u.squeeze(0).cpu().numpy()

    def select_action(self, state, deterministic=True):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor.sample(s, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def store(self, state, z, u, reward, cost, done):
        self.buf.append((np.asarray(state, np.float32), z, u, float(reward), float(cost), float(done)))

    def ready(self):
        return len(self.buf) >= self.steps_per_update

    # ------------------------------------------------------------ learning
    def _logp(self, s, z, u):
        mean, std, logits = self._dist(s)
        lp_c = (-0.5 * ((z - mean) / (std + EPS)) ** 2 - torch.log(std + EPS) - 0.5 * np.log(2 * np.pi)).sum(1)
        lp_d = torch.gather(F.log_softmax(logits, -1), -1, u.unsqueeze(-1)).squeeze(-1).sum(1)
        return lp_c + lp_d

    def _kl(self, s, old):
        mean, std, logits = self._dist(s)
        m0, s0, l0 = old
        kl_c = (torch.log(std / s0) + (s0**2 + (m0 - mean) ** 2) / (2 * std**2) - 0.5).sum(1)
        p0 = F.softmax(l0, -1)
        kl_d = (p0 * (F.log_softmax(l0, -1) - F.log_softmax(logits, -1))).sum(-1).sum(1)
        return (kl_c + kl_d).mean()

    def _gae(self, rew, val, done, gamma, lam):
        adv = np.zeros_like(rew)
        last = 0.0
        for t in reversed(range(len(rew))):
            nonterm = 1.0 - done[t]
            nxt = val[t + 1] if t + 1 < len(rew) else 0.0
            delta = rew[t] + gamma * nxt * nonterm - val[t]
            last = delta + gamma * lam * nonterm * last
            adv[t] = last
        return adv, adv + val

    def update(self):
        S = torch.as_tensor(np.stack([b[0] for b in self.buf]), device=self.device)
        Z = torch.as_tensor(np.stack([b[1] for b in self.buf]), device=self.device)
        U = torch.as_tensor(np.stack([b[2] for b in self.buf]), device=self.device, dtype=torch.long)
        rew = np.array([b[3] for b in self.buf], np.float32)
        cost = np.array([b[4] for b in self.buf], np.float32)
        done = np.array([b[5] for b in self.buf], np.float32)
        done[-1] = 1.0
        self.buf = []
        with torch.no_grad():
            v = self.vf(S).cpu().numpy()
            vc = self.vc(S).cpu().numpy()
        adv_r, ret_r = self._gae(rew, v, done, self.gamma, self.lam)
        adv_c, ret_c = self._gae(cost, vc, done, self.cgamma, self.clam)
        adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + EPS)
        adv_c = (1.0 - self.cgamma) * (adv_c - adv_c.mean())
        Ar = torch.as_tensor(adv_r, device=self.device)
        Ac = torch.as_tensor(adv_c, device=self.device)
        with torch.no_grad():
            m0, s0, l0 = self._dist(S)
            old = (m0, s0, l0)
            logp_old = self._logp(S, Z, U)
        J_c = float(cost.mean())
        c = J_c - self.cost_limit

        def surrogates():
            ratio = torch.exp(self._logp(S, Z, U) - logp_old)
            return (ratio * Ar).mean(), (ratio * Ac).mean()

        lr_, lc_ = surrogates()
        g = flat_grad(lr_, self.actor, retain_graph=True)
        b = flat_grad(lc_, self.actor)

        def Hvp(vec):
            kl = self._kl(S, old)
            gk = flat_grad(kl, self.actor, create_graph=True)
            return flat_grad(torch.dot(gk, vec), self.actor) + self.damping * vec

        v_g = conjugate_gradient(Hvp, g, self.cg_iters)
        delta = self.target_kl
        w = conjugate_gradient(Hvp, b, self.cg_iters)
        step, solution = solve_cpo_step(g, b, v_g, w, c, delta)
        case, nu = solution["case"], solution["nu"]
        theta0 = flat_params(self.actor)
        with torch.no_grad():
            lr0, lc0 = surrogates()
        accepted = False
        for k in range(self.backtrack_iters):
            set_flat_params(self.actor, theta0 + (self.backtrack_coeff**k) * step)
            with torch.no_grad():
                kl = self._kl(S, old).item()
                lr1, lc1 = surrogates()
            improve = (lr1 - lr0).item()
            dcost = (lc1 - lc0).item()
            ok_cost = dcost <= max(-c, 0.0) if case > 0 else dcost <= 0.0
            ok_rew = improve >= -1e-8 if c <= 0 and case > 0 else True
            if np.isfinite(kl) and kl <= delta + 1e-8 and ok_cost and ok_rew:
                accepted = True
                break
        if not accepted:
            set_flat_params(self.actor, theta0)
        self.lambda_val = float(nu)
        # value functions
        Rr = torch.as_tensor(ret_r, device=self.device)
        Rc = torch.as_tensor(ret_c, device=self.device)
        n = len(Rr)
        for _ in range(self.vf_iters):
            idx = torch.randint(0, n, (min(self.vf_batch, n),), device=self.device)
            for net, opt, tgt in ((self.vf, self.vf_opt, Rr), (self.vc, self.vc_opt, Rc)):
                loss = F.mse_loss(net(S[idx]), tgt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        return {"cpo_case": case, "nu": float(nu), "accepted": float(accepted), "J_c": J_c}
