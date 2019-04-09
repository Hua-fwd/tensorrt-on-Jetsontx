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

class PrintDims : public IPlugin
{
public:
	int cpySize;
	PrintDims() {		std::cout << "PrintDims" << std::endl;}
	PrintDims(const void* buffer, size_t size)
	{
		assert(size == sizeof(int));
		cpySize = *(reinterpret_cast<const int*>(buffer));
	}

	int getNbOutputs() const override
	{
		return 1;
	}
	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
	{
		std::cout << "PrintDims getOutputDimensions" << std::endl;
		//std::cout << "index: " << index << std::endl;
		//std::cout << "nbInputDims: " << nbInputDims << std::endl;
		assert(nbInputDims == 1);
		//assert(index <= 1);
		assert(inputs[0].nbDims == 3);
		for (int i = 0; i < nbInputDims; i++)
		{
			for (int j = 0; j < inputs[i].nbDims; j++)
			{
				std::cout << inputs[i].d[j] << ' ';
			}
			std::cout << std::endl;
		}
		cpySize = inputs[0].d[0]*inputs[0].d[1]*inputs[0].d[2];

		return inputs[0];
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
		  std::cout << "print enqueue" << std::endl;
		  std::cout << "size= " << cpySize << std::endl;
		  float *data = new float[cpySize];
		  newCHECK(cudaMemcpyAsync(data, inputs[0], cpySize * sizeof(float), cudaMemcpyDeviceToHost, stream));
		  for(int i=0; i < 100; i++) 
			{
				std::cout << data[i] << " ";
				if((i+1)%10==0) std::cout << std::endl;
			}
		  //std::cout << std::endl;
		std::cout << data[cpySize-1] << std::endl;
		newCHECK(cudaMemcpyAsync(outputs[0], inputs[0], cpySize * sizeof(float), cudaMemcpyDeviceToDevice, stream));
		//outputs[0] = const_cast<void *>(&(inputs[0]));
		delete[] data;
		return 0;
	}

	size_t getSerializationSize() override
	{
		return sizeof(int);
	}

	void serialize(void* buffer) override
	{
		*(reinterpret_cast<int *>(buffer)) = cpySize;
	}

	void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int)	override
	{
	}

};
