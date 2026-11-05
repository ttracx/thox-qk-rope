#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    ops.def("qk_norm(Tensor x, Tensor weight, float eps) -> Tensor");
    ops.def("apply_rope(Tensor x, Tensor cos, Tensor sin, int heads) -> Tensor");
    ops.def(
        "qk_norm_rope(Tensor q, Tensor k, Tensor q_weight, Tensor k_weight, Tensor cos, Tensor sin, float eps, int q_heads, int k_heads) -> (Tensor, Tensor)");
#if defined(CPU_KERNEL)
    ops.impl("qk_norm", torch::kCPU, &qk_norm_cpu);
    ops.impl("apply_rope", torch::kCPU, &apply_rope_cpu);
    ops.impl("qk_norm_rope", torch::kCPU, &qk_norm_rope_cpu);
#elif defined(CUDA_KERNEL)
    ops.impl("qk_norm", torch::kCUDA, &qk_norm_cuda);
    ops.impl("apply_rope", torch::kCUDA, &apply_rope_cuda);
    ops.impl("qk_norm_rope", torch::kCUDA, &qk_norm_rope_cuda);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
