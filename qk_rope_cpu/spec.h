#ifndef THOX_QK_ROPE_SPEC_H
#define THOX_QK_ROPE_SPEC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* fp32 contract, row-major.
 * qk_norm: y = weight * cast_fp32(rsqrt(mean(x^2)+eps) * x). cast is a no-op.
 * rope: NeoX rotate-half on the first `rotary` columns. cos/sin are
 * per token, shared across `heads`. rows = tokens * heads.
 * Partial rotary copies the tail unchanged.
 */

void thox_qk_norm_f32(
    const float* x,
    const float* weight,
    float* y,
    int64_t rows,
    int64_t dim,
    float eps);

void thox_apply_rope_f32(
    const float* x,
    const float* cos,
    const float* sin,
    float* y,
    int64_t tokens,
    int64_t heads,
    int64_t dim,
    int64_t rotary);

#ifdef __cplusplus
}
#endif

#endif
