"""Numerical specification for thox-qk-rope v1.

Clean-room implementation of two public definitions:

* RMSNorm over the last dimension, with the Qwen3RMSNorm promotion rule
  ``weight * normed.to(input_dtype)`` (Qwen Team, arXiv:2505.09388;
  the operator origin is Zhang and Sennrich, arXiv:1910.07467).
* NeoX rotate-half rotary embedding (Su et al., arXiv:2104.09864),
  the layout used by Qwen2, Qwen3, and Llama.

This module is the contract. Device kernels must match it. It does not
compute RoPE frequencies: the caller owns theta, ABF/YaRN scaling, and
the duplicated cos/sin tables.
"""

from __future__ import annotations

import numpy as np

__all__ = ["apply_rope", "layout_heads", "qk_norm", "qk_norm_rope"]


def _as_array(x):
    a = np.asarray(x)
    if not np.issubdtype(a.dtype, np.floating):
        raise TypeError(f"expected a floating dtype, got {a.dtype}")
    return a


def layout_heads(x_shape, cos_shape) -> int:
    """Head count for the device kernel's per-token cos/sin table.

    Returns ``x.shape[-2]`` when cos is one rank shorter and matches the
    token prefix (Qwen3 ``[B, S, H, D]`` with ``[B, S, R]``). Otherwise 1,
    which covers a table that already broadcasts onto every row.
    """
    x_shape = tuple(x_shape)
    cos_shape = tuple(cos_shape)
    if (
        len(x_shape) >= 2
        and len(cos_shape) + 1 == len(x_shape)
        and x_shape[:-2] == cos_shape[:-1]
    ):
        return int(x_shape[-2])
    return 1


def qk_norm(x, weight, eps: float = 1e-6):
    """Per-head RMSNorm. ``x`` is [..., D], ``weight`` is [D]."""
    x = _as_array(x)
    w = _as_array(weight)
    if w.ndim != 1 or w.shape[0] != x.shape[-1]:
        raise ValueError(f"weight shape {w.shape} does not match last dim {x.shape[-1]}")
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and >= 0")
    in_dtype = x.dtype
    xf = x.astype(np.float32, copy=False)
    variance = np.mean(np.square(xf), axis=-1, keepdims=True)
    normed = xf * np.reciprocal(np.sqrt(variance + np.float32(eps)))
    cast = normed.astype(in_dtype, copy=False)
    out_dtype = np.result_type(w.dtype, cast.dtype)
    return (w.astype(out_dtype, copy=False) * cast.astype(out_dtype, copy=False)).astype(
        out_dtype, copy=False
    )


def _align_table(x, table, name: str):
    """Broadcast ``table`` [..., R] onto ``x`` [..., D].

    Supported layouts:

    * ``table`` broadcasts onto ``x.shape[:-1] + (R,)``.
    * ``x`` is ``[..., heads, D]`` and ``table`` broadcasts onto
      ``x.shape[:-2] + (R,)`` (Qwen3 pre-transpose: cos is per token,
      shared by every head).
    """
    table = _as_array(table)
    if table.ndim < 1:
        raise ValueError(f"{name} must have a last dimension")
    rotary = int(table.shape[-1])
    if rotary < 2 or rotary % 2 != 0:
        raise ValueError(f"{name} rotary dim must be even and >= 2, got {rotary}")
    if rotary > x.shape[-1]:
        raise ValueError(f"{name} rotary dim {rotary} exceeds head dim {x.shape[-1]}")
    target = x.shape[:-1] + (rotary,)
    try:
        return np.broadcast_to(table, target), rotary
    except ValueError:
        pass
    if x.ndim >= 2:
        token_target = x.shape[:-2] + (rotary,)
        try:
            token = np.broadcast_to(table, token_target)
        except ValueError as exc:
            raise ValueError(
                f"{name} shape {table.shape} is not broadcastable to {target} "
                f"or to per-token {token_target}"
            ) from exc
        return np.broadcast_to(np.expand_dims(token, axis=-2), target), rotary
    raise ValueError(f"{name} shape {table.shape} is not broadcastable to {target}")


def rotate_half(x):
    half = x.shape[-1] // 2
    first = x[..., :half]
    second = x[..., half:]
    return np.concatenate((-second, first), axis=-1)


def apply_rope(x, cos, sin):
    """NeoX rotate-half RoPE. Math in fp32. Output dtype is result_type(x, cos)."""
    x = _as_array(x)
    if x.shape[-1] < 2:
        raise ValueError("head dim must be >= 2")
    cos_b, rotary = _align_table(x, cos, "cos")
    sin_b, rotary_s = _align_table(x, sin, "sin")
    if rotary != rotary_s:
        raise ValueError(f"cos rotary dim {rotary} != sin rotary dim {rotary_s}")
    rot = x[..., :rotary].astype(np.float32, copy=False)
    tail = x[..., rotary:]
    out_rot = rot * cos_b.astype(np.float32, copy=False)
    out_rot = out_rot + rotate_half(rot) * sin_b.astype(np.float32, copy=False)
    out_dtype = np.result_type(x.dtype, _as_array(cos).dtype)
    out_rot = out_rot.astype(out_dtype, copy=False)
    if tail.shape[-1] == 0:
        return out_rot
    return np.concatenate((out_rot, tail.astype(out_dtype, copy=False)), axis=-1)


def qk_norm_rope(q, k, q_weight, k_weight, cos, sin, eps: float = 1e-6):
    """Qwen3 order: RMSNorm on the head dim, then RoPE. Q and K may differ in heads."""
    qn = qk_norm(q, q_weight, eps)
    kn = qk_norm(k, k_weight, eps)
    return apply_rope(qn, cos, sin), apply_rope(kn, cos, sin)
