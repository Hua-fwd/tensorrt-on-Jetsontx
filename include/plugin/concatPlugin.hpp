#include <cassert>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
//#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <memory>
#include <cstring>
//#include <algorithm>
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

template<int CatAxis>
class Concat : public IPlugin
{
public:
	Concat() {}
	Concat(const void* buffer, size_t size)
	{
		//std::cout << "Concat" << std::endl;
		const char* d = reinterpret_cast<const char*>(buffer), *a = d;
		top_concat_dims = read<int>(d);
		num_concats_ = read<int>(d);
		concat_input_size_ = read<int>(d);
		mNbInputs = read<int>(d);
		/*std::cout << top_concat_dims << std::endl;
		std::cout << num_concats_ << std::endl;
		std::cout << mNbInputs << std::endl;*/
		bottom_concat_dims = new int[mNbInputs];
		
		for(int i = 0; i < mNbInputs; i++)
		{
			bottom_concat_dims[i] = read<int>(d);
			//std::cout << bottom_concat_dims[i] << std::endl;
		}
		
		assert(d == a + size);
		//std::cout << "Concat done" << std::endl;
	}

	int getNbOutputs() const override
	{
		return 1;
	}
	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
	{
		std::cout << "cancat getOutputDimensions" << std::endl;
		//std::cout << "index: " << index << std::endl;
		//std::cout << "nbInputDims: " << nbInputDims << std::endl;
		assert(nbInputDims == 6);
		assert(index <= 1);
		assert(inputs[index].nbDims == 3);
		for (int i = 1; i < nbInputDims; i++)
			for (int j = 0; j < inputs[index].nbDims; j++)
			{
				if(j == CatAxis) continue;
				assert(inputs[0].d[j] == inputs[i].d[j]);
			}
		
		DimsCHW dims(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
		for (int i = 1; i < nbInputDims; i++)
		{
			dims.d[CatAxis] += inputs[i].d[CatAxis];
			std::cout << dims.d[CatAxis] << " " <<  inputs[i].d[CatAxis] << std::endl;
		}

		std::cout << dims.d[0] << std::endl;
		std::cout << dims.d[1] << std::endl;
		std::cout << dims.d[2] << std::endl;
		return dims;
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

	// currently it is not possible for a plugin to execute "in place". Therefore we memcpy the data from the input to the output buffer
	int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override
	{
		//std::cout << "Concat" << std::endl;
		assert(batchSize == 1);
		//if (mNbInputs == 1) { return; }
		std::cout << "Concat enqueue" << std::endl;
		clock_t start_clock = clock();
		std::cout <<  clock() << std::endl;
		float* top_data = reinterpret_cast<float*>(outputs[0]);
		//float* top_data = new float[num_concats_*top_concat_dims*concat_input_size_];
		//float* bottom_data = new float[num_concats_*top_concat_dims*concat_input_size_];
		
		int offset_concat_axis = 0;
		//const int top_concat_dims = top[0]->shape(concat_axis_);
		for (int i = 0; i < mNbInputs; ++i) {
			const float* bottom_data = reinterpret_cast<const float*>(inputs[i]);
			//newCHECK(cudaMemcpyAsync(bottom_data, inputs[i], num_concats_ * bottom_concat_dims[i] * concat_input_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream));
			//const int bottom_concat_dims = bottom[i]->shape(concat_axis_);
			for (int n = 0; n < num_concats_; ++n) {
				//std::cout << "caffe_copy" << n << std::endl;
				caffe_copy(bottom_concat_dims[i] * concat_input_size_,
				  bottom_data + n * bottom_concat_dims[i] * concat_input_size_,
				  top_data + (n * top_concat_dims + offset_concat_axis)
					  * concat_input_size_, false/* gpuModel*/);
				//std::cout << "caffe_copy done" << n << std::endl;
			}
			offset_concat_axis += bottom_concat_dims[i];
		}
		//newCHECK(cudaMemcpyAsync(outputs[0], top_data, num_concats_*top_concat_dims*concat_input_size_ * sizeof(float), cudaMemcpyHostToDevice, stream));
		//delete[] top_data;
		//delete[] bottom_data;
  	  clock_t clock_sum = 0;
	  clock_sum += (clock() - start_clock);
		std::cout <<  clock() << std::endl;
	  std::cout <<  static_cast<double>( clock_sum ) / CLOCKS_PER_SEC << std::endl;
		return 0;
	}

	size_t getSerializationSize() override
	{
		return sizeof(int)*(4+mNbInputs);
	}

	void serialize(void* buffer) override
	{
		char* d = reinterpret_cast<char*>(buffer), *a = d;

		write(d, top_concat_dims);
		write(d, num_concats_);
		write(d, concat_input_size_);
		write(d, mNbInputs);
		
		for(int i = 0; i < mNbInputs; i++)
		{
			write(d, bottom_concat_dims[i]);
			//std::cout << bottom_concat_dims[i] << std::endl;
		}


		assert(d == a + getSerializationSize());
	}

	void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int)	override
	{
		mNbInputs = nbInputs;
		bottom_concat_dims = new int[nbInputs];
		for(int i = 0; i < CatAxis; i++)
		{
			num_concats_ *= inputs[0].d[i];
		}
		for(int i = CatAxis+1; i < inputs[0].nbDims; i++)
		{
			concat_input_size_ *= inputs[0].d[i];
		}
		
		top_concat_dims = outputs[0].d[CatAxis];
		for(int i = 0; i < nbInputs; i++)
		{
			bottom_concat_dims[i] = inputs[i].d[CatAxis];
		}
	}

	~Concat()
	{
		delete[] bottom_concat_dims;
	}

private:
	int *bottom_concat_dims = NULL, //bottom dims of CatAxis
	top_concat_dims = 1, //top dim of CatAxis
	num_concats_ = 1, //copy times
	concat_input_size_ = 1; //copy size
	
protected:	
	int mNbInputs;

	template<typename T> void write(char*& buffer, const T& val)
	{
		*reinterpret_cast<T*>(buffer) = val;
		buffer += sizeof(T);
	}

	template<typename T> T read(const char*& buffer)
	{
		T val = *reinterpret_cast<const T*>(buffer);
		buffer += sizeof(T);
		return val;
	}

};
