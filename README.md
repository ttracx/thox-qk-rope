---
license: apache-2.0
tags:
  - thox
  - kernel-source
  - qk-norm
  - rope
  - local-first
  - inference
---

# thox-qk-rope

Query-key RMSNorm and NeoX rotary embedding for local decode. Source and a tested numerical contract. Not a Kernel Hub binary.

**Your AI. Your Data. Your Rules.**

Source git: [ttracx/thox-qk-rope](https://github.com/ttracx/thox-qk-rope). Hub source mirror: [Thox-ai/thox-qk-rope-src](https://huggingface.co/Thox-ai/thox-qk-rope-src).

| Item | State |
|---|---|
| What this repository is | kernel-builder source plus the v1 reference |
| Kernel Hub binary | Not uploaded. `get_kernel("Thox-ai/thox-qk-rope")` will not resolve until a `kernel-builder` upload |
| Reference tests | NumPy contract and the fp32 C spec. No device timing |
| Backends in `build.toml` | CUDA and CPU, float32. Metal is a shader note only |
| Companion kernel | [`Thox-ai/thox-edge-ops`](https://huggingface.co/kernels/Thox-ai/thox-edge-ops) (residual RMSNorm and SiLU-and-mul). This repository does not modify it |

`thox-edge-ops` covers the pre-norm residual and the SwiGLU gate. It does not apply QK-norm or RoPE. Qwen3 adds per-head QK-norm before RoPE. Qwen2.5 needs RoPE and does not use QK-norm. Those are the two calls this kernel exists to serve. Gemma's `(1 + weight)` norm is out of scope, as it already is for `thox-edge-ops`.

## Contract

```text
y = weight * dtype(x)(x * rsqrt(mean(x^2) + eps))     # Qwen3RMSNorm promotion
out = rotate_half(y) with the caller-supplied cos, sin
```

The caller owns the frequency table (`theta`, ABF, YaRN, partial rotary). This kernel does not invent positions.

```python
import sys
sys.path.insert(0, "torch-ext")
from thox_qk_rope import apply_rope, qk_norm_rope
```

Layouts: `[..., D]` with a broadcastable cos/sin table, or Qwen3's pre-transpose view `[batch, seq, heads, dim]` with cos/sin `[batch, seq, rotary]`. `rotary` is even and may be shorter than `dim`; the tail is copied after the norm. v1 native kernels are float32. The reference also states the float16 promotion rule and is what a later half-precision backend must match.

## Build, when a machine with Nix is available

`flake.lock` is not vendored. On a build host:

```bash
nix flake update
kernel-builder build-and-copy -L
```

Upload to the kernel hub, creating `kernels/Thox-ai/thox-qk-rope` rather than this source tree, is `kernel-builder upload` after that lock exists. Do not point `get_kernel` at a source snapshot.

## Tests

```bash
python3 -m unittest tests.test_reference tests.test_spec -v
```

## Paper

`paper/main.tex` is the technical note. It reports no speedup.

Apache-2.0. Copyright 2026 THOX.ai LLC.
