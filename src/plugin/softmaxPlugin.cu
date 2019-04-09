#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "plugin/softmaxPlugin.hpp"
#include "util/math_functions.hpp"


__global__ void kernel_channel_max(const int num, const int channels,
    const int spatial_dim, const float* data, float* out) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    float maxval = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
      maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
    }
    out[index] = maxval;
  }
}

__global__ void kernel_channel_subtract(const int count,
    const int num, const int channels,
    const int spatial_dim, const float* channel_max, float* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] -= channel_max[n * spatial_dim + s];
  }
}

__global__ void kernel_exp(const int count, const float* data, float* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = exp(data[index]);
  }
}

__global__ void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, const float* data, float* channel_sum) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    float sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}

__global__ void kernel_channel_div(const int count,
    const int num, const int channels,
    const int spatial_dim, const float* channel_sum, float* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] /= channel_sum[n * spatial_dim + s];
  }
}

__global__ void kernel_channel_dot(const int num, const int channels,
    const int spatial_dim, const float* data_1, const float* data_2,
    float* channel_dot) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    float dot = 0;
    for (int c = 0; c < channels; ++c) {
      dot += (data_1[(n * channels + c) * spatial_dim + s]
          * data_2[(n * channels + c) * spatial_dim + s]);
    }
    channel_dot[index] = dot;
  }
}

template <int Axis>
void Softmax<Axis>::Forward_gpu(int batchSize, const void*const *inputs, void** outputs, void* workspace, cudaStream_t stream) {
  const float* bottom_data = reinterpret_cast<const float*>(inputs[0]);
  int dataDim = dims[0]*dims[1]*dims[2];
  float* top_data = reinterpret_cast<float*>(outputs[0]);
  //float* scale_data = scale_.mutable_gpu_data();
  float* scale_data;
  newCHECK(cudaMalloc(&scale_data, scaleDim * sizeof(float)));
  int count = dataDim;
  int channels = dims[Axis];
  LOG(INFO) << "0";
  caffe_copy(count, bottom_data, top_data, false);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  // compute max
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_max<<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_, top_data,
      scale_data);
  // subtract
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_subtract<<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, outer_num_, channels, inner_num_,
      scale_data, top_data);
  // exponentiate
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_exp<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_data, top_data);
  // sum after exp
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_sum<<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_, top_data,
      scale_data);
  // divide
  // NOLINT_NEXT_LINE(whitespace/operators)
  kernel_channel_div<<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, outer_num_, channels, inner_num_,
      scale_data, top_data);
  newCHECK(cudaFree(scale_data));
}

