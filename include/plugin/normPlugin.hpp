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

#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "util/math_functions.hpp"
#include "common.hpp"
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

#define CHECK(status)												\
    {																\
	if (status != 0)												\
	{																\
	    std::cout << "Cuda failure: " << status;					\
	    abort();													\
	}																\
    }



class NormPlugin: public IPlugin
{
private:
	//float* buffer_data, *norm_data, *sum_channel_multiplier;

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

	Weights copyToDevice(const void* hostData, size_t count)
	{
		void* deviceData;
		newCHECK(cudaMalloc(&deviceData, count * sizeof(float)));
		newCHECK(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
		return Weights{ DataType::kFLOAT, deviceData, int64_t(count) };
	}

	void serializeFromDevice(char*& hostBuffer, Weights deviceWeights)
	{		
		cudaMemcpy(hostBuffer, deviceWeights.values, deviceWeights.count * sizeof(float), cudaMemcpyDeviceToHost);
		hostBuffer += deviceWeights.count * sizeof(float);
	}

	Weights deserializeToDevice(const char*& hostBuffer, size_t count)
	{
		Weights w = copyToDevice(hostBuffer, count);
		hostBuffer += count * sizeof(float);
		return w;	
	}

	int mNbChannels, mNbWidth, mNbHeight;
	cudnnHandle_t mCudnn;
	cublasHandle_t mCublas;
	//Weights mKernelWeights, mBiasWeights;
	cudnnTensorDescriptor_t mSrcDescriptor, mDstDescriptor;
	bool across_spatial_;
	bool channel_shared_;
	float eps_;
	Weights mScalesWeights;
	
public:
	NormPlugin(const Weights *weights, int nbWeights): across_spatial_(false), channel_shared_(false), eps_(1e-10)
	{
		// since we want to deal with the case where there is no bias, we can't infer
		// the number of channels from the bias weights.

		assert(nbWeights == 1);
		mScalesWeights = copyToDevice(weights[0].values, weights[0].count);
		//assert(mBiasWeights.count == 0 || mBiasWeights.count == nbOutputChannels);
		//mKernelWeights = copyToDevice(weights[0].values, weights[0].count);
		//mBiasWeights = copyToDevice(weights[1].values, weights[1].count);
		//assert(mBiasWeights.count == 0 || mBiasWeights.count == nbOutputChannels);

		//mNbInputChannels = int(weights[0].count / nbOutputChannels);
	}
	// NormPlugin():across_spatial_(false), channel_shared_(false), eps_(1e-10) {}
	// create the plugin at runtime from a byte stream
	/* NormPlugin(const void* data, size_t length)
	{
		const char* d = reinterpret_cast<const char*>(data), *a = d;
		mNbInputChannels = read<int>(d);
		mNbOutputChannels = read<int>(d);
		int biasCount = read<int>(d);

		mKernelWeights = deserializeToDevice(d, mNbInputChannels * mNbOutputChannels);
		mBiasWeights = deserializeToDevice(d, biasCount);
		assert(d == a + length);
	} */
	
	NormPlugin(const void* data, size_t length): across_spatial_(false), channel_shared_(false), eps_(1e-10)
	{
		const char* d = reinterpret_cast<const char*>(data), *a = d;
		mNbChannels = read<int>(d);
		mNbWidth = read<int>(d);
		mNbHeight = read<int>(d);
		//int biasCount = read<int>(d);

		mScalesWeights = deserializeToDevice(d, mNbChannels);
		assert(d == a + length);
	}

	void initParams();

	~NormPlugin()
	{
		cudaFree(const_cast<void*>(mScalesWeights.values));
		/*cudaFree(reinterpret_cast<void*>(buffer_data));
		cudaFree(reinterpret_cast<void*>(norm_data));
		cudaFree(reinterpret_cast<void*>(sum_channel_multiplier));*/
		//cudaFree(const_cast<void*>(mBiasWeights.values));
	}

	int getNbOutputs() const override
	{
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
	{
		assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
		//assert(mNbInputChannels == inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2]);

		return DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
	}

	void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override
	{
		mNbChannels = inputDims[0].d[0];
		mNbWidth = inputDims[0].d[1];
		mNbHeight = inputDims[0].d[2];
	}

	int initialize() override
	{
		CHECK(cudnnCreate(&mCudnn));							// initialize cudnn and cublas
		CHECK(cublasCreate(&mCublas));
		CHECK(cudnnCreateTensorDescriptor(&mSrcDescriptor));	// create cudnn tensor descriptors we need for bias addition
		CHECK(cudnnCreateTensorDescriptor(&mDstDescriptor));

		return 0;
	}

	virtual void terminate() override
	{
		CHECK(cublasDestroy(mCublas));
		CHECK(cudnnDestroy(mCudnn));
	}

	virtual size_t getWorkspaceSize(int maxBatchSize) const override
	{
		return 0;
	}
	
	void Forward_cpu(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
	{
		float kONE = 1.0f;//, kZERO = 0.0f;
		//cublasSetStream(mCublas, stream);
		//cudnnSetStream(mCudnn, stream);
		
		//const float* bottom_data = reinterpret_cast<const float*>(inputs[0]);
	    //float* top_data = reinterpret_cast<float*>(outputs[0]);
	    //const float* scale = reinterpret_cast<const float*>(mScalesWeights.values);
		std::cout << "NormPlugin enqueue" << std::endl;

	    int num = batchSize;
	    int dim = mNbChannels * mNbHeight * mNbWidth;
	    int spatial_dim = mNbHeight * mNbWidth;
	    int channels = mNbChannels;
		float* bottom_data = new float[num*dim];
		float* top_data = new float[num*dim];
		float* scale = new float[mScalesWeights.count];
		newCHECK(cudaMemcpyAsync(bottom_data, inputs[0], num * dim * sizeof(float), cudaMemcpyDeviceToHost, stream));
		newCHECK(cudaMemcpyAsync(scale, mScalesWeights.values, mScalesWeights.count * sizeof(float), cudaMemcpyDeviceToHost, stream));

		float* buffer_data = new float[dim];
		//caffe_copy<float>(dim, bottom_data, buffer_data);
		
		float* sum_channel_multiplier = new float[channels];
	    float* sum_spatial_multiplier = new float[spatial_dim];
		float* norm_data = new float[spatial_dim];

		caffe_set<float>(channels, kONE, sum_channel_multiplier);
		caffe_set<float>(spatial_dim, kONE, sum_spatial_multiplier);
	    // add eps to avoid overflow
		caffe_set<float>(spatial_dim, float(eps_), norm_data);
		
		for (int n = 0; n < num; ++n) {
			bottom_data += dim*n;
			top_data += dim*n;
			caffe_sqr<float>(dim, bottom_data, buffer_data);
			if (across_spatial_) {
				  // add eps to avoid overflow
				  norm_data[n] = pow(caffe_cpu_asum<float>(dim, buffer_data)+eps_,
									 float(0.5));
				  caffe_cpu_scale<float>(dim, float(1.0 / norm_data[n]), bottom_data,
										 top_data);
			} else {
					// norm_data = buffer_data' * sum_channel_multiplier + norm_data
					//(spatial_dim)(spatial_dim, channels) (channels, 1) (spatial_dim)
				  caffe_cpu_gemv<float>(CblasTrans, channels, spatial_dim, float(1),
										buffer_data, sum_channel_multiplier, float(1),
										norm_data);
				  // compute norm
				  // norm_data = norm_data .^ 0.5
				  caffe_powx<float>(spatial_dim, norm_data, float(0.5), norm_data);
				  // scale the layer
				  // buffer_data = sum_channel_multiplier * norm_data
				  // (channels, spatial_dim)  (channels, 1) (1, spatial_dim)
				  caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
										1, float(1), sum_channel_multiplier, norm_data,
										float(0), buffer_data);
				  caffe_div<float>(dim, bottom_data, buffer_data, top_data);
				  norm_data += spatial_dim;
			}
			// scale the output
			if (channel_shared_) {
				caffe_scal<float>(dim, scale[0], top_data);
			} else {
				caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
									1, float(1), scale, sum_spatial_multiplier,
									float(0),
									buffer_data);
					
				caffe_mul<float>(dim, top_data, buffer_data, top_data);
			}
		 }
		/*newCHECK(cublasSgemm(mCublas, CUBLAS_OP_T, CUBLAS_OP_N, mNbOutputChannels, batchSize, mNbInputChannels, &kONE, 
				reinterpret_cast<const float*>(mKernelWeights.values), mNbInputChannels, 
				reinterpret_cast<const float*>(inputs[0]), mNbInputChannels, &kZERO, 
				reinterpret_cast<float*>(outputs[0]), mNbOutputChannels));
		if (mBiasWeights.count)
		{
			newCHECK(cudnnSetTensor4dDescriptor(mSrcDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, mNbOutputChannels, 1, 1));
			newCHECK(cudnnSetTensor4dDescriptor(mDstDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, mNbOutputChannels, 1, 1));
			newCHECK(cudnnAddTensor(mCudnn, &kONE, mSrcDescriptor, mBiasWeights.values, &kONE, mDstDescriptor, outputs[0]));
		}*/
		//newCHECK(cudaMemcpyAsync(buffers[inputIndex0], inputData, batchSize * dataDim * sizeof(float), cudaMemcpyHostToDevice, stream));
		newCHECK(cudaMemcpyAsync(outputs[0], top_data, num * dim * sizeof(float), cudaMemcpyHostToDevice, stream));
		delete[] bottom_data;
		delete[] top_data;
		delete[] scale;
		delete[] buffer_data;
		delete[] sum_channel_multiplier;
	    delete[] sum_spatial_multiplier;
	}

	virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override
	{
		std::cout << "NormPlugin enqueue" << std::endl;
		clock_t start_clock = clock();
		std::cout <<  clock() << std::endl;
	
		/*std::cout << "1" << std::endl; 
		cublasHandle_t mCublas = Caffe::cublas_handle();
		std::cout << "2" << std::endl; 
		cublasSetStream(mCublas, stream);
		std::cout << "3" << std::endl; */
		cublasSetStream(mCublas, stream);
		cudnnSetStream(mCudnn, stream);
		//initParams();
		Forward_gpu(batchSize,inputs, outputs, workspace, stream);

		clock_t clock_sum = 0;
	    clock_sum += (clock() - start_clock);
	    std::cout <<  clock() << std::endl;
	    std::cout <<  static_cast<double>( clock_sum ) / CLOCKS_PER_SEC << std::endl;
		/*float kONE = 1.0f;//, kZERO = 0.0f;
		cublasSetStream(mCublas, stream);
		cudnnSetStream(mCudnn, stream);
		
		const float* bottom_data = reinterpret_cast<const float*>(inputs[0]);
	    float* top_data = reinterpret_cast<float*>(outputs[0]);
	    const float* scale = reinterpret_cast<const float*>(mScalesWeights.values);


	    int num = batchSize;
	    int dim = mNbChannels * mNbHeight * mNbWidth;
	    int spatial_dim = mNbHeight * mNbWidth;
	    int channels = mNbChannels;
		
		float* buffer_data = new float[dim];
		//caffe_copy<float>(dim, bottom_data, buffer_data);
		
		float* sum_channel_multiplier = new float[channels];
	    float* sum_spatial_multiplier = new float[spatial_dim];
		float* norm_data = new float[spatial_dim];

		caffe_set<float>(channels, kONE, sum_channel_multiplier);
		caffe_set<float>(spatial_dim, kONE, sum_spatial_multiplier);
	    // add eps to avoid overflow
		caffe_set<float>(spatial_dim, float(eps_), norm_data);
		
		for (int n = 0; n < num; ++n) {
			caffe_sqr<float>(dim, bottom_data, buffer_data);
			if (across_spatial_) {
				  // add eps to avoid overflow
				  norm_data[n] = pow(caffe_cpu_asum<float>(dim, buffer_data)+eps_,
									 float(0.5));
				  caffe_cpu_scale<float>(dim, float(1.0 / norm_data[n]), bottom_data,
										 top_data);
			} else {
					// norm_data = buffer_data' * sum_channel_multiplier + norm_data
					//(spatial_dim)(spatial_dim, channels) (channels, 1) (spatial_dim)
				  caffe_cpu_gemv<float>(CblasTrans, channels, spatial_dim, float(1),
										buffer_data, sum_channel_multiplier, float(1),
										norm_data);
				  // compute norm
				  // norm_data = norm_data .^ 0.5
				  caffe_powx<float>(spatial_dim, norm_data, float(0.5), norm_data);
				  // scale the layer
				  // buffer_data = sum_channel_multiplier * norm_data
				  // (channels, spatial_dim)  (channels, 1) (1, spatial_dim)
				  caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
										1, float(1), sum_channel_multiplier, norm_data,
										float(0), buffer_data);
				  caffe_div<float>(dim, bottom_data, buffer_data, top_data);
				  norm_data += spatial_dim;
			}
			// scale the output
			if (channel_shared_) {
				caffe_scal<float>(dim, scale[0], top_data);
			} else {
				caffe_cpu_gemm<float>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
									1, float(1), scale, sum_spatial_multiplier,
									float(0),
									buffer_data);
				caffe_mul<float>(dim, top_data, buffer_data, top_data);
			}
			bottom_data += dim;
			top_data += dim;
		 }*/
		/*newCHECK(cublasSgemm(mCublas, CUBLAS_OP_T, CUBLAS_OP_N, mNbOutputChannels, batchSize, mNbInputChannels, &kONE, 
				reinterpret_cast<const float*>(mKernelWeights.values), mNbInputChannels, 
				reinterpret_cast<const float*>(inputs[0]), mNbInputChannels, &kZERO, 
				reinterpret_cast<float*>(outputs[0]), mNbOutputChannels));
		if (mBiasWeights.count)
		{
			newCHECK(cudnnSetTensor4dDescriptor(mSrcDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, mNbOutputChannels, 1, 1));
			newCHECK(cudnnSetTensor4dDescriptor(mDstDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, mNbOutputChannels, 1, 1));
			newCHECK(cudnnAddTensor(mCudnn, &kONE, mSrcDescriptor, mBiasWeights.values, &kONE, mDstDescriptor, outputs[0]));
		}*/
		return 0;
	}

	/* virtual size_t getSerializationSize() override
	{
		// 3 integers (number of input channels, number of output channels, bias size), and then the weights:
		return sizeof(int)*3 + mKernelWeights.count*sizeof(float) + mBiasWeights.count*sizeof(float);
	}

	virtual void serialize(void* buffer) override
	{
		char* d = reinterpret_cast<char*>(buffer), *a = d;

		write(d, mNbInputChannels);
		write(d, mNbOutputChannels);
		write(d, (int)mBiasWeights.count);
		serializeFromDevice(d, mKernelWeights);
		serializeFromDevice(d, mBiasWeights);

		assert(d == a + getSerializationSize());
	} */
	
	virtual size_t getSerializationSize() override
	{
		// 3 integers (number of input channels, number of output channels, bias size), and then the weights:
		return sizeof(int)*3 + mScalesWeights.count*sizeof(float);
	}

	virtual void serialize(void* buffer) override
	{
		char* d = reinterpret_cast<char*>(buffer), *a = d;

		write(d, mNbChannels);
		write(d, mNbHeight);
		write(d, mNbWidth);
		serializeFromDevice(d, mScalesWeights);

		assert(d == a + getSerializationSize());
	}

	void Forward_gpu(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream);

};
