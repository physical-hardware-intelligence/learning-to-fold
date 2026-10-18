"""Shrink camera frames on the robot computer, the way the policy server would.

The server resizes every image to 224x224 itself, with openpi's `resize_with_pad`
(openpi c23745b, src/openpi/shared/image_tools.py): `jax.image.resize(method=LINEAR)`, which
antialiases when shrinking, then round to uint8, then pad the short side with black.
Sending full frames costs 6.14 MB per request; sending the shrunk picture costs 0.41 MB.

Doing it here only works if the result is the same image. Ilia found the policy can tell how a
frame was resized (report section 9.1: a two-step resize lowered sim success), so this file
re-implements that exact operation: the same separable triangle-kernel weights as
`jax/_src/image/scale.py::compute_weight_mat`, in float32, applied as two matrix products.

Checked 2026-10-18 on 300 real frames from the organizer dataset (100 per camera), against
openpi's function run in JAX 0.5.3 on CPU: identical except 75 to 500 of 15,052,800 values per
camera that differ by 1 (of 255). The server's own resize of an already-224 image, or of the
picture without its padding, returns it unchanged. Check against the server's GPU: pending.
"""
from __future__ import annotations

import numpy as np

_CACHE: dict[tuple[int, int, int, int], tuple[np.ndarray, np.ndarray]] = {}


def _weights(n_in: int, n_out: int) -> np.ndarray:
    """(n_in, n_out) weights of jax.image.resize, linear kernel, antialias=True, translation 0."""
    scale = np.float32(n_out / n_in)
    inv = np.float32(1.0) / scale
    kernel_scale = max(inv, np.float32(1.0))
    sample = (np.arange(n_out, dtype=np.float32) + np.float32(0.5)) * inv - np.float32(0.5)
    x = np.abs(sample[None, :] - np.arange(n_in, dtype=np.float32)[:, None]) / kernel_scale
    w = np.maximum(np.float32(0.0), np.float32(1.0) - x)
    total = w.sum(axis=0, keepdims=True)
    w = np.where(np.abs(total) > 1000.0 * np.finfo(np.float32).eps, w / np.where(total != 0, total, 1), 0)
    inside = (sample >= -0.5) & (sample <= n_in - 0.5)
    return np.where(inside[None, :], w, 0).astype(np.float32)


def _shape(h: int, w: int, height: int, width: int) -> tuple[int, int]:
    ratio = max(w / width, h / height)               # same expressions as openpi, same rounding
    return int(h / ratio), int(w / ratio)


def shrink(img: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
    """uint8 (h, w, 3) -> uint8 (rh, rw, 3): the picture part of resize_with_pad, without the padding."""
    h, w, c = img.shape
    rh, rw = _shape(h, w, height, width)
    key = (h, w, rh, rw)
    if key not in _CACHE:
        _CACHE[key] = (np.ascontiguousarray(_weights(h, rh).T), _weights(w, rw))
    wh_t, ww = _CACHE[key]
    x = wh_t @ img.reshape(h, w * c).astype(np.float32)          # rows: (rh, w*c)
    x = x.reshape(rh, w, c).transpose(0, 2, 1) @ ww              # columns: (rh, c, rw)
    return np.clip(np.round(x.transpose(0, 2, 1)), 0, 255).astype(np.uint8)


def resize_with_pad(img: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
    """The full openpi result: shrink, then black padding centred as openpi centres it."""
    x = shrink(img, height, width)
    rh, rw = x.shape[:2]
    ph0, rem_h = divmod(height - rh, 2)
    pw0, rem_w = divmod(width - rw, 2)
    return np.pad(x, ((ph0, ph0 + rem_h), (pw0, pw0 + rem_w), (0, 0)))
