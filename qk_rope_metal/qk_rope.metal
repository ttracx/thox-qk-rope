// Metal shader for the same fp32 contract as qk_rope_cpu/spec.c.
// Not listed in build.toml. Wiring a host and an MPS registration is a
// separate Mac kernel-builder task. Do not describe this file as a
// shipped Metal kernel.

#include <metal_stdlib>
using namespace metal;

kernel void thox_qk_norm_f32(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant int& dim [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
    device const float* in = x + row * dim;
    device float* out = y + row * dim;
    float sumsq = 0.f;
    for (uint i = lane; i < (uint)dim; i += lanes) {
        float v = in[i];
        sumsq += v * v;
    }
    threadgroup float partial[32];
    if (lane < 32) {
        partial[lane] = 0.f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Single-simdgroup reduction is enough when the host launches 32 threads.
    sumsq = simd_sum(sumsq);
    if (lane == 0) {
        partial[0] = sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = rsqrt(partial[0] / float(dim) + eps);
    for (uint i = lane; i < (uint)dim; i += lanes) {
        out[i] = (in[i] * inv_rms) * weight[i];
    }
}
