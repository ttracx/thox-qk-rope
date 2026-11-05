#include "spec.h"

#include <math.h>

void thox_qk_norm_f32(
    const float* x,
    const float* weight,
    float* y,
    int64_t rows,
    int64_t dim,
    float eps) {
    const float inv_dim = 1.0f / (float)dim;
    for (int64_t row = 0; row < rows; ++row) {
        const float* in = x + row * dim;
        float* out = y + row * dim;
        float sumsq = 0.0f;
        for (int64_t i = 0; i < dim; ++i) {
            float v = in[i];
            sumsq += v * v;
        }
        float inv_rms = 1.0f / sqrtf(sumsq * inv_dim + eps);
        for (int64_t i = 0; i < dim; ++i) {
            out[i] = (in[i] * inv_rms) * weight[i];
        }
    }
}

void thox_apply_rope_f32(
    const float* x,
    const float* cos,
    const float* sin,
    float* y,
    int64_t tokens,
    int64_t heads,
    int64_t dim,
    int64_t rotary) {
    const int64_t half = rotary / 2;
    for (int64_t token = 0; token < tokens; ++token) {
        const float* c = cos + token * rotary;
        const float* s = sin + token * rotary;
        for (int64_t head = 0; head < heads; ++head) {
            int64_t row = token * heads + head;
            const float* in = x + row * dim;
            float* out = y + row * dim;
            for (int64_t i = 0; i < half; ++i) {
                float a = in[i];
                float b = in[half + i];
                out[i] = a * c[i] - b * s[i];
                out[half + i] = b * c[half + i] + a * s[half + i];
            }
            for (int64_t i = rotary; i < dim; ++i) {
                out[i] = in[i];
            }
        }
    }
}
