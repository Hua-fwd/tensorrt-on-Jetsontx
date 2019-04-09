#include <cassert>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <memory>
#include <cstring>
#include <algorithm>
#include <ctime>

#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "util/math_functions.hpp"
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
using namespace caffe;

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


template<int Axis>
class Softmax : public IPlugin
{
public:
	Softmax() {}
	Softmax(const void* buffer, size_t size)
	{
		assert(size == sizeof(int)*3);
		const int* d = reinterpret_cast<const int*>(buffer);
		dims[0] = d[0];
		dims[1] = d[1];
		dims[2] = d[2];
		/*std::cout << "Softmax" << std::endl;

		std::cout << dims[0] << std::endl;
		std::cout << dims[1] << std::endl;
		std::cout << dims[2] << std::endl;*/
		
		initParams();  
		
		//assert(dims[1] == 8);
		assert(dims[2] == 1);
		assert(Axis == 1);
	}

	int getNbOutputs() const override
	{
		return 1;
	}
	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
	{
		//std::cout << "index: " << index << std::endl;
		//std::cout << "nbInputDims: " << nbInputDims << std::endl;
		assert(nbInputDims == 1);
		assert(index == 0);
		assert(Axis <= inputs[0].nbDims);
		assert(inputs[0].nbDims == 3);
		/*for (int i = 1; i < inputs[index].nbDims; i++)
		{
			if(i == CatAxis) continue;
			assert(inputs[0].d[i] == inputs[index].d[i]);
		}
		
		DimsCHW dims(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
		for (int i = 1; i < nbInputDims; i++)
		{
			dims.d[CatAxis] += inputs[i].d[CatAxis];
		}*/
		return DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
	}

	int initialize() override
	{
		return 0;
	}

	void terminate() override
	{
	}

	size_t getWorkspaceSize(int) const override
	{
		return 0;
	}
	
	void Forward_cpu(int batchSize, const void*const *inputs, void** outputs, void* workspace, cudaStream_t stream)
	{
		int dataDim = dims[0]*dims[1]*dims[2];
		float* bottom_data = new float[dataDim];
		float* top = new float[dataDim];
		float* top_data = top;

		newCHECK(cudaMemcpyAsync(bottom_data, inputs[0], dataDim * batchSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
		/*std::cout << "softmax enqueue" << std::endl;
		for(int i=0; i < 100; i++) std::cout << bottom_data[i] << " ";
		std::cout << std::endl;*/
		//float* scale_data = scale_.mutable_cpu_data();
		float* multiplier_data = new float[dims[Axis]];
		caffe_set(dims[Axis], 1.0f, multiplier_data);
		float* scale_data = new float[scaleDim];
		int channels = dims[Axis];
		int dim = dataDim / outer_num_;
		caffe_copy(dataDim, bottom_data, top_data);
		// We need to subtract the max to avoid numerical issues, compute the exp,
		// and then normalize.
		  for (int i = 0; i < outer_num_; ++i) {
			// initialize scale_data to the first plane
			caffe_copy(inner_num_, bottom_data + i * dim, scale_data);
			for (int j = 0; j < channels; j++) {
			  for (int k = 0; k < inner_num_; k++) {
				scale_data[k] = std::max(scale_data[k],
					bottom_data[i * dim + j * inner_num_ + k]);
			  }
			}
			// subtraction
			caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, channels, inner_num_,
				1, -1., multiplier_data, scale_data, 1., top_data);
			// exponentiation
			caffe_exp<float>(dim, top_data, top_data);
			// sum after exp
			caffe_cpu_gemv<float>(CblasTrans, channels, inner_num_, 1.,
				top_data, multiplier_data, 0., scale_data);
			// division
			for (int j = 0; j < channels; j++) {
			  caffe_div(inner_num_, top_data, scale_data, top_data);
			  top_data += inner_num_;
			}
		  }
		/*std::cout << "softmax enqueue" << std::endl;
		for(int i=0; i < 100; i++) std::cout << top[i] << " ";
		std::cout << std::endl;
		std::cout << top[dataDim-1] << std::endl;*/
		newCHECK(cudaMemcpyAsync(outputs[0], top, dataDim * batchSize * sizeof(float), cudaMemcpyHostToDevice, stream));
		delete[] bottom_data;
		delete[] top;
		delete[] scale_data;
		delete[] multiplier_data;
	}
		
	void Forward_gpu(int batchSize, const void*const *inputs, void** outputs, void* workspace, cudaStream_t stream);

	// currently it is not possible for a plugin to execute "in place". Therefore we memcpy the data from the input to the output buffer
	int enqueue(int batchSize, const void*const *inputs, void** outputs, void* workspace, cudaStream_t stream) override
	{
		assert(batchSize == 1);
		std::cout << "softmax enqueue" << std::endl;
		clock_t start_clock = clock();
		std::cout <<  clock() << std::endl;
		Forward_gpu(batchSize, inputs, outputs, workspace, stream);
  		clock_t clock_sum = 0;
		clock_sum += (clock() - start_clock);
		std::cout <<  clock() << std::endl;
		std::cout <<  static_cast<double>( clock_sum ) / CLOCKS_PER_SEC << std::endl;

		return 0;
	}

	size_t getSerializationSize() override
	{
		return sizeof(int)*(3);
	}

	void serialize(void* buffer) override
	{
		int* d = reinterpret_cast<int*>(buffer);
		
		d[0] = dims[0];
		d[1] = dims[1];
		d[2] = dims[2];

		/*write(d, top_concat_dims);
		write(d, num_concats_);
		write(d, concat_input_size_);
		write(d, mNbInputs);
		
		for(int i = 0; i < mNbInputs; i++)
		{
			write(d, bottom_concat_dims[i]);
		}*/


		//assert(d == a + getSerializationSize());
	}

	void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int)	override
	{
		  //std::cout << "Softmax configure" << std::endl;
		  //top[0]->ReshapeLike(*bottom[0]);
		  //vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));
		  //sum_multiplier_.Reshape(mult_dims);
		  //float* multiplier_data = sum_multiplier_.mutable_cpu_data();
		  dims[0] = inputs[0].d[0];
		  dims[1] = inputs[0].d[1];
		  dims[2] = inputs[0].d[2];
		  /*std::cout << dims[0] << std::endl;
		  std::cout << dims[1] << std::endl;
		  std::cout << dims[2] << std::endl;*/

		  
		  //outer_num_ = bottom[0]->count(0, softmax_axis_);
		  //inner_num_ = bottom[0]->count(softmax_axis_ + 1);
		  
		  //vector<int> scale_dims = bottom[0]->shape();
		  //scale_dims[softmax_axis_] = 1;
		  //scale_.Reshape(scale_dims);
	}
	void initParams()
	{
		//softmax_axis_ = Axis;
		//multiplier_data = new float[dims[Axis]];
		//caffe_set(dims[Axis], 1.0f, multiplier_data);
		//std::cout << "Axis " << Axis << std::endl;
		//std::cout << "for init outer_num_" << outer_num_ << std::endl;
		outer_num_ = 1;
		for(int i = 0; i < Axis; i++)
		{
			outer_num_ *= dims[i];
			/*std::cout << "i " << i << std::endl;
			std::cout << "dims " << dims[i] << std::endl;
			std::cout << "outer_num_ " << outer_num_ << std::endl;*/
		}
		//std::cout << "outer_num_ " << outer_num_ << std::endl;
		
		//assert inputs[0].nbInputDims = 3
		inner_num_ = 1;
		for(int i = Axis+1; i < 3; i++)
		{
			inner_num_ *= dims[i];
			/*std::cout << "i " << i << std::endl;
			std::cout << "dims " << dims[i] << std::endl;
			std::cout << "inner_num_ " << inner_num_ << std::endl;*/
		}
		//std::cout << "inner_num_ " << inner_num_ << std::endl;

		scaleDim = 1;
		for(int i = 0; i < 3; i++)
		{
			if(i == Axis) continue;
			scaleDim *= dims[i];
		}
		//scale_data = new float[scaleDim];
		//std::cout << "scaleDim" << scaleDim << std::endl;
	}

private:
	int outer_num_, inner_num_, dims[3], scaleDim;
	//float *multiplier_data;//, *scale_data;
	
protected:	

};

template class Softmax<1>;
