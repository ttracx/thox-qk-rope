"""Load the reference and run one Qwen3-shaped forward step. No timings."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "torch-ext"))

from thox_qk_rope import apply_rope, qk_norm_rope

rng = np.random.default_rng(0)
batch, seq, heads, kv_heads, dim = 1, 1, 8, 2, 16
q = rng.standard_normal((batch, seq, heads, dim)).astype(np.float32)
k = rng.standard_normal((batch, seq, kv_heads, dim)).astype(np.float32)
wq = np.ones(dim, dtype=np.float32)
wk = np.ones(dim, dtype=np.float32)
pos = np.arange(seq, dtype=np.float32)[:, None]
freq = np.arange(dim // 2, dtype=np.float32)
angles = pos * np.exp(-np.log(10000.0) * freq / (dim // 2))
cos_h = np.cos(angles).astype(np.float32)
sin_h = np.sin(angles).astype(np.float32)
cos = np.concatenate([cos_h, cos_h], axis=-1)
sin = np.concatenate([sin_h, sin_h], axis=-1)
cos = np.broadcast_to(cos, (batch, seq, dim)).copy()
sin = np.broadcast_to(sin, (batch, seq, dim)).copy()

qn, kn = qk_norm_rope(q, k, wq, wk, cos, sin)
rope_only = apply_rope(q, cos, sin)
print("q", tuple(qn.shape), qn.dtype, "k", tuple(kn.shape), "rope", tuple(rope_only.shape))
