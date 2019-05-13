#include <torch/extension.h>
#include <iostream>

#define CHECK_TYPE(x, y) AT_ASSERTM(x.type() == y.type(), "Must be of the same type")
#define CHECK_FLOAT(x) AT_ASSERTM(x.dtype() == torch::kFloat32, "Must be float32 type")
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), "Must be CUDA tensors")
#define CHECK_DIMENSION(x, y) AT_ASSERTM((x) == (y), "Dimensions of tensors must be equal")

void cuda_l1_t(float *c, const float *a, const float *b, const int M, const int N, const int D);

torch::Tensor l1_t(const torch::Tensor &a, const torch::Tensor &b)
{
    CHECK_TYPE(a, b);
    CHECK_FLOAT(a);
    CHECK_CUDA(a);

    auto a_sizes = a.sizes(), b_sizes = b.sizes();
    CHECK_DIMENSION(a_sizes[1], b_sizes[1]);
    
    int m = a_sizes[0], n = b_sizes[0], d = a_sizes[1];
    torch::Tensor c = at::empty({m, n}, a.type());

    float *a_ptr = a.data<float>(), *b_ptr = b.data<float>(), *c_ptr = c.data<float>();
    cuda_l1_t(c_ptr, a_ptr, b_ptr, m, n, d);
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("l1_t", &l1_t);
}
