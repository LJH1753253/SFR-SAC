import numpy as np
import torch

HYBRID_AP_MODES = {"hybrid_ap_gumbel", "hybrid_ap_onehot"}
VALID_AP_PROJECTION_MODES = {"legacy", "center_snap"} | HYBRID_AP_MODES


def _validate_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode not in VALID_AP_PROJECTION_MODES:
        raise ValueError(
            f"Unknown ap_projection_mode={mode}, expected one of {sorted(VALID_AP_PROJECTION_MODES)}"
        )
    return mode


def ap_centers_numpy(num_rsu: int) -> np.ndarray:
    num_rsu = int(num_rsu)
    if num_rsu <= 1:
        return np.zeros(1, dtype=np.float32)
    return np.linspace(-1.0, 1.0, num_rsu, dtype=np.float32)


def ap_centers_torch(num_rsu: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    num_rsu = int(num_rsu)
    if num_rsu <= 1:
        return torch.zeros(1, device=device, dtype=dtype)
    return torch.linspace(-1.0, 1.0, num_rsu, device=device, dtype=dtype)


def snap_ap_values_numpy(a_ap: np.ndarray, num_rsu: int, mode: str = "legacy") -> np.ndarray:
    mode = _validate_mode(mode)
    a_ap = np.asarray(a_ap, dtype=np.float32)
    if mode == "legacy" or int(num_rsu) <= 1:
        return np.clip(a_ap, -1.0, 1.0).astype(np.float32, copy=False)
    centers = ap_centers_numpy(num_rsu)
    clipped = np.clip(a_ap, -1.0, 1.0)
    flat = clipped.reshape(-1)
    idx = np.abs(flat[:, None] - centers[None, :]).argmin(axis=1)
    snapped = centers[idx].reshape(clipped.shape)
    return snapped.astype(np.float32, copy=False)


def snap_ap_values_torch(
    a_ap: torch.Tensor, num_rsu: int, mode: str = "legacy", *, straight_through: bool = False
) -> torch.Tensor:
    mode = _validate_mode(mode)
    if mode == "legacy" or int(num_rsu) <= 1:
        return a_ap.clamp(-1.0, 1.0)
    centers = ap_centers_torch(num_rsu, device=a_ap.device, dtype=a_ap.dtype)
    clipped = a_ap.clamp(-1.0, 1.0)
    dist = torch.abs(clipped.unsqueeze(-1) - centers.view(*[1] * clipped.dim(), -1))
    idx = dist.argmin(dim=-1)
    snapped = centers[idx]
    if straight_through:
        snapped = clipped + (snapped - clipped).detach()
    return snapped


def project_action_numpy(
    action: np.ndarray, *, num_devices: int, num_rsu: int, mode: str = "legacy"
) -> np.ndarray:
    mode = _validate_mode(mode)
    arr = np.asarray(action, dtype=np.float32).copy()
    n = int(num_devices)
    r = int(num_rsu)
    if mode in HYBRID_AP_MODES:
        expected = 2 * n + n * r
        if arr.shape[-1] < expected:
            return arr
        arr[: 2 * n] = np.clip(arr[: 2 * n], -1.0, 1.0)
        ap = arr[2 * n : expected].reshape(n, r)
        idx = np.argmax(ap, axis=1)
        onehot = np.zeros((n, r), dtype=np.float32)
        onehot[np.arange(n), idx] = 1.0
        arr[2 * n : expected] = onehot.reshape(-1)
        return arr
    if arr.shape[-1] < 3 * n or mode == "legacy" or int(num_rsu) <= 1:
        return arr
    arr[2 * n : 3 * n] = snap_ap_values_numpy(arr[2 * n : 3 * n], num_rsu, mode=mode)
    return arr


def project_action_torch(
    action: torch.Tensor,
    *,
    num_devices: int,
    num_rsu: int,
    mode: str = "legacy",
    straight_through: bool = False,
) -> torch.Tensor:
    mode = _validate_mode(mode)
    n = int(num_devices)
    r = int(num_rsu)
    if mode in HYBRID_AP_MODES:
        expected = 2 * n + n * r
        if action.shape[-1] < expected:
            return action
        prefix = action[..., : 2 * n].clamp(-1.0, 1.0)
        ap = action[..., 2 * n : expected].reshape(*action.shape[:-1], n, r)
        idx = ap.argmax(dim=-1)
        onehot = torch.nn.functional.one_hot(idx, num_classes=r).to(
            dtype=action.dtype, device=action.device
        )
        if straight_through:
            onehot = ap + (onehot - ap).detach()
        onehot = onehot.reshape(*action.shape[:-1], n * r)
        suffix = action[..., expected:]
        parts = [prefix, onehot]
        if suffix.shape[-1] > 0:
            parts.append(suffix)
        return torch.cat(parts, dim=-1)
    if action.shape[-1] < 3 * n or mode == "legacy" or int(num_rsu) <= 1:
        return action
    prefix = action[..., : 2 * n]
    a_ap = action[..., 2 * n : 3 * n]
    snapped_ap = snap_ap_values_torch(a_ap, num_rsu, mode=mode, straight_through=straight_through)
    suffix = action[..., 3 * n :]
    parts = [prefix, snapped_ap]
    if suffix.shape[-1] > 0:
        parts.append(suffix)
    return torch.cat(parts, dim=-1)


def a_ap_to_rsu_index(a_ap: np.ndarray, num_rsu: int, mode: str = "legacy") -> np.ndarray:
    mode = _validate_mode(mode)
    if int(num_rsu) <= 1:
        return np.zeros_like(np.asarray(a_ap, dtype=np.float32), dtype=np.int64)
    snapped = snap_ap_values_numpy(a_ap, num_rsu, mode=mode)
    x = (np.clip(snapped, -1.0, 1.0) + 1.0) / 2.0
    idx = np.rint(x * (int(num_rsu) - 1)).astype(np.int64)
    return np.clip(idx, 0, int(num_rsu) - 1)
