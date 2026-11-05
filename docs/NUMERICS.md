# Numerics v1

## RMSNorm

For a row `x` with width `D` and a weight `w` of width `D`:

```text
inv = rsqrt(mean_i float32(x_i)^2 + eps)
n_i = float32(x_i) * inv
y = w * cast_to_dtype(x)(n)
```

The product uses ordinary dtype promotion (`result_type(w, cast(n))`). That is the rule in the Qwen3 RMSNorm module: `weight * hidden.to(input_dtype)`. It is not the "cast, then multiply, then cast again" rule used by some other THOX norm notes. The two kernels are different operators. Do not mix their golden values.

`eps` default in the Python reference is `1e-6`.

A zero row stays zero for any finite weight.

## RoPE

NeoX rotate-half on the first `R` elements, `R` even, `R <= D`:

```text
out[:R/2]     = a * cos[:R/2] - b * sin[:R/2]
out[R/2:R]    = b * cos[R/2:] + a * sin[R/2:]
out[R:]       = tail
```

where `a = x[:R/2]` and `b = x[R/2:R]`. The multiply is done in float32. The stored dtype is `result_type(x.dtype, cos.dtype)`.

Cos and sin are the tables the caller would have passed to a rotate-half embedding, including the duplicated half that Hugging Face builds for Qwen and Llama. This kernel does not expand a half-length frequency vector.

If `x` is `[..., heads, D]` and cos is `[..., R]` with one fewer rank, every head of a token shares that token's table. Any other shape must broadcast onto `[..., R]`.

## Fused order

`qk_norm_rope` is norm, then RoPE. It is not RoPE, then norm. Q and K have independent weights and may have different head counts. They share one cos/sin table.

## What v1 has actually been executed

- The NumPy reference, including the float16 promotion cases, against an independent oracle.
- `qk_rope_cpu/spec.c` in float32, compiled locally and compared with the reference.

Not executed: the Torch CPU binding, the CUDA kernels, the Metal shader, any GPU, any end-to-end model, any tokens per second.
