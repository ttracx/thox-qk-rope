#pragma once

#include <torch/torch.h>

#include <tuple>

torch::Tensor qk_norm_cpu(torch::Tensor x, torch::Tensor weight, double eps);
torch::Tensor apply_rope_cpu(torch::Tensor x, torch::Tensor cos, torch::Tensor sin, int64_t heads);
std::tuple<torch::Tensor, torch::Tensor> qk_norm_rope_cpu(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor cos,
    torch::Tensor sin,
    double eps,
    int64_t q_heads,
    int64_t k_heads);

torch::Tensor qk_norm_cuda(torch::Tensor x, torch::Tensor weight, double eps);
torch::Tensor apply_rope_cuda(torch::Tensor x, torch::Tensor cos, torch::Tensor sin, int64_t heads);
std::tuple<torch::Tensor, torch::Tensor> qk_norm_rope_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor q_weight,
    torch::Tensor k_weight,
    torch::Tensor cos,
    torch::Tensor sin,
    double eps,
    int64_t q_heads,
    int64_t k_heads);
