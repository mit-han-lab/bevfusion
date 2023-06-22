#include <stdio.h>
#include <stdlib.h>

// __global__ void feature_decorator_kernel(int b, int d, int h, int w, int n, int c, int n_intervals,
//                                   const float *__restrict__ x,
//                                   const int *__restrict__ geom_feats,
//                                   const int *__restrict__ interval_starts,
//                                   const int *__restrict__ interval_lengths,
//                                   float* __restrict__ out) {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   int index = idx / c;
//   int cur_c = idx % c;
//   if (index >= n_intervals) return;
//   int interval_start = interval_starts[index];
//   int interval_length = interval_lengths[index];
//   const int* cur_geom_feats = geom_feats + interval_start * 4;
//   const float* cur_x = x + interval_start * c + cur_c;
//   float* cur_out = out + cur_geom_feats[3] * d * h * w * c + 
//     cur_geom_feats[2] * h * w * c + cur_geom_feats[0] * w * c + 
//     cur_geom_feats[1] * c + cur_c;
//   float psum = 0;
//   for(int i = 0; i < interval_length; i++){
//     psum += cur_x[i * c];
//   }
//   *cur_out = psum;
// }

__global__ void feature_decorator_kernel(float* __restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // if (idx == 0) {
  //   out = 5.0;
  // }
  out[0] = 5.0;
  out[15] = 6.0;
}



// void feature_decorator(int b, int d, int h, int w, int n, int c, int n_intervals, const float* x,
//   const int* geom_feats, const int* interval_starts, const int* interval_lengths, float* out) {
//   feature_decorator_kernel<<<(int)ceil(((double)n_intervals * c / 256)), 256>>>(
//     b, d, h, w, n, c, n_intervals, x, geom_feats, interval_starts, interval_lengths, out
//   );
// }

void feature_decorator(float* out) {
  feature_decorator_kernel<<<1, 1>>>(out);
}

