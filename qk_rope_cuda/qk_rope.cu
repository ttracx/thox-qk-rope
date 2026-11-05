// thox-qk-rope CUDA. fp32 path. Indexing matches qk_rope_cpu/spec.c.
// One warp per row. Head dim up to 256 is the supported launch shape;
// larger rows are rejected by the binding.

#include <torch/torch.h>

#include <cuda_runtime.h>

namespace {

constexpr int kMaxDim = 256;

__device__ float warp_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    return v;
}

__global__ void qk_norm_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ y,
    int dim,
    float eps) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    const float* in = x + static_cast<long long>(row) * dim;
    float* out = y + static_cast<long long>(row) * dim;
    float sumsq = 0.f;
    for (int i = lane; i < dim; i += 32) {
        float v = in[i];
        sumsq += v * v;
    }
    sumsq = warp_sum(sumsq);
    const float inv_rms = rsqrtf(sumsq / static_cast<float>(dim) + eps);
    for (int i = lane; i < dim; i += 32) {
        out[i] = (in[i] * inv_rms) * weight[i];
    }
}

__global__ void apply_rope_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    float* __restrict__ y,
    int heads,
    int dim,
    int rotary) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    const int token = row / heads;
    const float* in = x + static_cast<long long>(row) * dim;
    float* out = y + static_cast<long long>(row) * dim;
    const float* c = cos + static_cast<long long>(token) * rotary;
    const float* s = sin + static_cast<long long>(token) * rotary;
    const int half = rotary / 2;
    for (int i = lane; i < half; i += 32) {
        float a = in[i];
        float b = in[half + i];
        out[i] = a * c[i] - b * s[i];
        out[half + i] = b * c[half + i] + a * s[half + i];
    }
    for (int i = rotary + lane; i < dim; i += 32) {
        out[i] = in[i];
    }
}

void check_fp32(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32 in v1");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor qk_norm_cuda(torch::Tensor x, torch::Tensor weight, double eps) {
    check_fp32(x, "x");
    check_fp32(weight, "weight");
    const int dim = static_cast<int>(x.size(-1));
    TORCH_CHECK(dim >= 1 && dim <= kMaxDim, "head dim must be in 1..", kMaxDim);
    TORCH_CHECK(weight.numel() == dim, "weight numel must equal head dim");
    const int rows = static_cast<int>(x.numel() / dim);
    auto y = torch::empty_like(x);
    qk_norm_f32_kernel<<<rows, 32>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), y.data_ptr<float>(), dim, static_cast<float>(eps));
    return y;
}

torch::Tensor apply_rope_cuda(torch::Tensor x, torch::Tensor cos, torch::Tensor sin, int64_t heads) {
    check_fp32(x, "x");
    check_fp32(cos, "cos");
    check_fp32(sin, "sin");
    const int dim = static_cast<int>(x.size(-1));
    const int rotary = static_cast<int>(cos.size(-1));
    TORCH_CHECK(heads >= 1, "heads must be >= 1");
    TORCH_CHECK(dim <= kMaxDim && rotary <= dim && rotary % 2 == 0, "bad rotary/head dim");
    TORCH_CHECK(sin.size(-1) == rotary, "sin rotary dim mismatch");
    const int rows = static_cast<int>(x.numel() / dim);
    TORCH_CHECK(rows % heads == 0, "row count is not divisible by heads");
    const int tokens = rows / static_cast<int>(heads);
    TORCH_CHECK(cos.numel() == static_cast<long long>(tokens) * rotary, "cos has the wrong number of tokens");
    TORCH_CHECK(sin.numel() == cos.numel(), "sin numel mismatch");
    auto y = torch::empty_like(x);
    apply_rope_f32_kernel<<<rows, 32>>>(
        x.data_ptr<float>(),
        cos.data_ptr<float>(),
        sin.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<int>(heads),
        dim,
        rotary);
    return y;
}

std::tuple<torch::Tensor, torch::Tensor> qk_norm_rope_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor cos,
    torch::Tensor sin,
    double eps,
    int64_t q_heads,
    int64_t k_heads) {
    auto qn = qk_norm_cuda(q, q_weight, eps);
    auto kn = qk_norm_cuda(k, k_weight, eps);
    return {apply_rope_cuda(qn, cos, sin, q_heads), apply_rope_cuda(kn, cos, sin, k_heads)};
}
