"""thox-qk-rope.

The NumPy reference is the v1 contract and the fallback. A kernel-builder
build injects ``_ops.py``; when that module is present and the inputs are
Torch tensors, calls go to the device op.
"""

from __future__ import annotations

from .reference import apply_rope as _ref_apply_rope
from .reference import layout_heads
from .reference import qk_norm as _ref_qk_norm
from .reference import qk_norm_rope as _ref_qk_norm_rope

__all__ = ["apply_rope", "layout_heads", "qk_norm", "qk_norm_rope"]

try:
    from ._ops import ops as _OPS
except ImportError:
    _OPS = None


def _torch_tensor(value):
    mod = type(value).__module__
    return mod == "torch" or mod.startswith("torch.")


def qk_norm(x, weight, eps: float = 1e-6):
    if _OPS is not None and _torch_tensor(x):
        return _OPS.qk_norm(x, weight, float(eps))
    return _ref_qk_norm(x, weight, eps)


def apply_rope(x, cos, sin):
    if _OPS is not None and _torch_tensor(x):
        heads = layout_heads(tuple(x.shape), tuple(cos.shape))
        return _OPS.apply_rope(x, cos, sin, int(heads))
    return _ref_apply_rope(x, cos, sin)


def qk_norm_rope(q, k, q_weight, k_weight, cos, sin, eps: float = 1e-6):
    if _OPS is not None and _torch_tensor(q):
        return _OPS.qk_norm_rope(
            q,
            k,
            q_weight,
            k_weight,
            cos,
            sin,
            float(eps),
            int(layout_heads(tuple(q.shape), tuple(cos.shape))),
            int(layout_heads(tuple(k.shape), tuple(cos.shape))),
        )
    return _ref_qk_norm_rope(q, k, q_weight, k_weight, cos, sin, eps)
