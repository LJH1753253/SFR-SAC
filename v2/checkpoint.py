"""Source identities and episode-boundary training checkpoints."""

from __future__ import annotations

import copy
import hashlib
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=8)
def source_digest(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    paths = sorted(
        path
        for folder in ("agents", "envs", "offloading", "utils", "v2", "cloud24")
        for path in (root / folder).glob("*.py")
    )
    paths += sorted((root / "configs").glob("*.json"))
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def agent_state(agent):
    modules, optimizers, scalars = {}, {}, {}
    for name, value in vars(agent).items():
        if isinstance(value, torch.nn.Module):
            modules[name] = value.state_dict()
        elif isinstance(value, torch.optim.Optimizer):
            optimizers[name] = value.state_dict()
        elif name in {"alpha", "lambda_val", "lam", "log_alpha", "update_step", "total_it", "select_it"}:
            scalars[name] = value.detach().clone() if torch.is_tensor(value) else value
    return dict(modules=modules, optimizers=optimizers, scalars=scalars,
                rollout=copy.deepcopy(getattr(agent, "buf", [])))


def restore_agent(agent, state):
    for name, weights in state["modules"].items():
        getattr(agent, name).load_state_dict(weights)
    for name, value in state["scalars"].items():
        current = getattr(agent, name)
        if torch.is_tensor(current):
            with torch.no_grad():
                current.copy_(value)
        else:
            setattr(agent, name, value)
    for name, weights in state["optimizers"].items():
        getattr(agent, name).load_state_dict(weights)
    if hasattr(agent, "buf"):
        agent.buf = state["rollout"]
