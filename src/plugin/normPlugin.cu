#include <algorithm>
#include <cfloat>
#include <vector>

//#include "thrust/device_vector.h"

//#include "caffe/filler.hpp"
#include "plugin/normPlugin.hpp"
#include "util/math_functions.hpp"
#include "util/device_alternate.hpp"

#define newCHECK(status)												\
    {																\
	if (status != 0)												\
	{																\
	    std::cout << "Cuda failure: " << cudaGetErrorString(status)	\
		      << " at line " << __LINE__							\
                      << std::endl;									\
	    abort();													\
	}																\
    }

// divid a matrix with vector
__global__ void DivBsx(const int nthreads, const float* A,
    const float* v, const int rows, const int cols, const CBLAS_TRANSPOSE trans,
    float* B) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int c = index % cols;
    int r = (index / cols) % rows;
    if (trans == CblasNoTrans) {
      B[index] = A[index] / v[c];
    } else {
      B[index] = A[index] / v[r];
    }
  }
}

__global__ void MulBsx(const int nthreads, const float* A,
    const float* v, const int rows, const int cols, const CBLAS_TRANSPOSE trans,
    float* B) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int c = index % cols;
    int r = (index / cols) % rows;
    if (trans == CblasNoTrans) {
      B[index] = A[index] * v[c];
    } else {
      B[index] = A[index] * v[r];
    }
  }
}

/*void NormPlugin::initParams()
{
	newCHECK(cudaMalloc(&sum_channel_multiplier, mNbChannels * sizeof(float)));
	newCHECK(cudaMalloc(&norm_data, mNbHeight * mNbWidth * sizeof(float)));
	newCHECK(cudaMalloc(&buffer_data, mNbHeight * mNbWidth * mNbChannels * sizeof(float)));	
	caffe_gpu_set<float>(mNbChannels, 1.0f, sum_channel_multiplier);
	if (!across_spatial_) caffe_gpu_set<float>(mNbHeight * mNbWidth, float(eps_), norm_data);
}*/

void NormPlugin::Forward_gpu(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) {
	float* buffer_data, *norm_data, *sum_channel_multiplier;
	newCHECK(cudaMalloc(&sum_channel_multiplier, mNbChannels * sizeof(float)));
	newCHECK(cudaMalloc(&norm_data, mNbHeight * mNbWidth * sizeof(float)));
	newCHECK(cudaMalloc(&buffer_data, mNbHeight * mNbWidth * mNbChannels * sizeof(float)));	
	void* buffer_data_dev = buffer_data, *norm_data_dev = norm_data, *sum_channel_multiplier_dev = sum_channel_multiplier;
	caffe_gpu_set<float>(mNbChannels, 1.0f, sum_channel_multiplier);
	if (!across_spatial_) caffe_gpu_set<float>(mNbHeight * mNbWidth, float(eps_), norm_data);

	int num = batchSize;
	int dim = mNbChannels * mNbHeight * mNbWidth;
	int spatial_dim = mNbHeight * mNbWidth;
	int channels = mNbChannels;
  const float* bottom_data = reinterpret_cast<const float*>(inputs[0]);
  float* top_data = reinterpret_cast<float*>(outputs[0]);

  const float* scale = reinterpret_cast<const float*>(mScalesWeights.values);
  if (across_spatial_) {
    // need to index it
    //norm_data = norm_;
  } else {
    //norm_data = norm_;
    // add eps to avoid overflow
    //caffe_gpu_set<float>(spatial_dim, float(eps_), norm_data);
  }
  //const float* scale;
  if (channel_shared_) {
    //scale = this->blobs_[0]->cpu_data();
  } else {
    //scale = this->blobs_[0]->gpu_data();
  }
  /*const float* sum_channel_multiplier = sum_channel_multiplier_;
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / num;
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  int channels = bottom[0]->channels();*/
  for (int n = 0; n < num; ++n) {
    caffe_gpu_powx<float>(dim, bottom_data, float(2), buffer_data);
    if (across_spatial_) {
      float normsqr;
      caffe_gpu_asum<float>(dim, buffer_data, &normsqr);
      // add eps to avoid overflow
      norm_data[n] = pow(normsqr+eps_, float(0.5));
      caffe_gpu_scale<float>(dim, float(1.0 / norm_data[n]), bottom_data,
                             top_data);
    } else {
      // compute norm
      caffe_gpu_gemv<float>(CblasTrans, channels, spatial_dim, float(1),
                            buffer_data, sum_channel_multiplier, float(1),
                            norm_data);
      caffe_gpu_powx<float>(spatial_dim, norm_data, float(0.5), norm_data);
      // scale the layer
      // NOLINT_NEXT_LINE(whitespace/operators)
      DivBsx<<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
          dim, bottom_data, norm_data, channels, spatial_dim, CblasNoTrans,
          top_data);
      CUDA_POST_KERNEL_CHECK;
      norm_data += spatial_dim;
    }
    // scale the output
    if (channel_shared_) {
      caffe_gpu_scal<float>(dim, scale[0], top_data);
    } else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      MulBsx<<<CAFFE_GET_BLOCKS(dim), CAFFE_CUDA_NUM_THREADS>>>(
          dim, top_data, scale, channels, spatial_dim, CblasTrans,
          top_data);
      CUDA_POST_KERNEL_CHECK;
    }
    bottom_data += dim;
    top_data += dim;
  }

	newCHECK(cudaFree((buffer_data_dev)));
	newCHECK(cudaFree((norm_data_dev)));
	newCHECK(cudaFree((sum_channel_multiplier_dev)));
}
