#include <torch/torch.h>

// CUDA function declarations
// void feature_decorator(int b, int d, int h, int w, int n, int c, int n_intervals, const float* x,
//     const int* geom_feats, const int* interval_starts, const int* interval_lengths, float* out);

void feature_decorator(float* out);

at::Tensor feature_decorator_forward(
  const at::Tensor _x, 
  const at::Tensor _y, 
  const at::Tensor _z, 
  const double vx, const double vy, const double x_offset, const double y_offset, 
  int normalize_coords, int use_cluster, int use_center
) {
  int n = _x.size(0);
  int c = _x.size(1);
  int a = _x.size(2);
  auto options = torch::TensorOptions().dtype(_x.dtype()).device(_x.device());
  int decorate_dims = 0;
  if (use_cluster > 0) {
    decorate_dims += 3;
  }
  if (use_center > 0) {
    decorate_dims += 2;
  }

  at::Tensor _out = torch::zeros({n, c, a+decorate_dims}, options);
  float* out = _out.data_ptr<float>();
  const float* x = _x.data_ptr<float>();
  const int* y = _y.data_ptr<int>();
  const int* z = _z.data_ptr<int>();
  feature_decorator(out);
  return _out;

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("feature_decorator_forward", &feature_decorator_forward,
        "feature_decorator_forward");
}

static auto registry =
    torch::RegisterOperators("feature_decorator_ext::feature_decorator_forward", &feature_decorator_forward);
