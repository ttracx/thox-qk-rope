---
library_name: kernels
license: apache-2.0
tags:
- kernels
- qk-norm
- rope
- edge-ai
- inference
---

# thox-qk-rope

Per-head query-key RMSNorm and NeoX rotary embedding for local decode.
The caller supplies cos and sin. The kernel does not compute frequencies.

| Op | Order | v1 native dtype |
|---|---|---|
| `qk_norm(x, weight, eps)` | RMSNorm on the last dimension, Qwen3 promotion | float32 |
| `apply_rope(x, cos, sin)` | NeoX rotate-half, optional partial rotary | float32 |
| `qk_norm_rope(q, k, wq, wk, cos, sin, eps)` | norm, then RoPE, on Q and on K | float32 |

CUDA and CPU are in `build.toml`. Metal is not. There is no backward pass.
No speed figure is published with this source.

## How to use

```python
from kernels import get_kernel

ops = get_kernel("{{ repo_id }}", version={{ version }})
q, k = ops.qk_norm_rope(q, k, q_weight, k_weight, cos, sin, 1e-6)
```

Shapes: `q` is `[batch, seq, heads, dim]`, `k` is `[batch, seq, kv_heads, dim]`,
`cos` and `sin` are `[batch, seq, rotary]` with `rotary` even and `rotary <= dim`.

## Available functions
{% if functions %}
{% for func in functions %}
- `{{ func }}`
{% endfor %}
{% endif %}

## Benchmarks
{% if has_benchmark %}
Run `kernels benchmark {{ repo_id }} --version {{ version }}`.
{% else %}
No benchmark is shipped. Do not treat a missing table as a performance claim.
{% endif %}

## Scope

- float32 only on the native path in this source
- NeoX rotate-half only. Interleaved GPT-J RoPE is not implemented
- Gemma `(1 + weight)` norms are not implemented
- Not a replacement for `Thox-ai/thox-edge-ops`
{% if source %}

## Source

{{ source }}
{% endif %}
