#include <torch/torch.h>

#include "spec.h"

namespace {

torch::Tensor norm_f32(torch::Tensor x, torch::Tensor weight, double eps) {
    TORCH_CHECK(x.is_cpu() && x.is_contiguous() && x.scalar_type() == torch::kFloat32, "x must be contiguous cpu float32");
    TORCH_CHECK(weight.is_cpu() && weight.is_contiguous() && weight.scalar_type() == torch::kFloat32, "weight must be contiguous cpu float32");
    const auto dim = x.size(-1);
    TORCH_CHECK(weight.numel() == dim, "weight numel must equal head dim");
    auto y = torch::empty_like(x);
    thox_qk_norm_f32(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        y.data_ptr<float>(),
        x.numel() / dim,
        dim,
        static_cast<float>(eps));
    return y;
}

torch::Tensor rope_f32(torch::Tensor x, torch::Tensor cos, torch::Tensor sin, int64_t heads) {
    TORCH_CHECK(x.is_cpu() && x.is_contiguous() && x.scalar_type() == torch::kFloat32, "x must be contiguous cpu float32");
    TORCH_CHECK(cos.is_cpu() && cos.is_contiguous() && cos.scalar_type() == torch::kFloat32, "cos must be contiguous cpu float32");
    TORCH_CHECK(sin.is_cpu() && sin.is_contiguous() && sin.scalar_type() == torch::kFloat32, "sin must be contiguous cpu float32");
    const auto dim = x.size(-1);
    const auto rotary = cos.size(-1);
    TORCH_CHECK(heads >= 1 && rotary % 2 == 0 && rotary <= dim, "bad heads or rotary dim");
    TORCH_CHECK(sin.size(-1) == rotary, "sin rotary dim mismatch");
    const auto rows = x.numel() / dim;
    TORCH_CHECK(rows % heads == 0, "row count is not divisible by heads");
    const auto tokens = rows / heads;
    TORCH_CHECK(cos.numel() == tokens * rotary && sin.numel() == cos.numel(), "cos/sin token count mismatch");
    auto y = torch::empty_like(x);
    thox_apply_rope_f32(
        x.data_ptr<float>(),
        cos.data_ptr<float>(),
        sin.data_ptr<float>(),
        y.data_ptr<float>(),
        tokens,
        heads,
        dim,
        rotary);
    return y;
}

}  // namespace

torch::Tensor qk_norm_cpu(torch::Tensor x, torch::Tensor weight, double eps) {
    return norm_f32(x, weight, eps);
}

torch::Tensor apply_rope_cpu(torch::Tensor x, torch::Tensor cos, torch::Tensor sin, int64_t heads) {
    return rope_f32(x, cos, sin, heads);
}

std::tuple<torch::Tensor, torch::Tensor> qk_norm_rope_cpu(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor cos,
    torch::Tensor sin,
    double eps,
    int64_t q_heads,
    int64_t k_heads) {
    auto qn = norm_f32(q, q_weight, eps);
    auto kn = norm_f32(k, k_weight, eps);
    return {rope_f32(qn, cos, sin, q_heads), rope_f32(kn, cos, sin, k_heads)};
}
