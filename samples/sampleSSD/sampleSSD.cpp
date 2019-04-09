#include <cassert>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <memory>
#include <cstring>
#include <algorithm>
#include <ctime>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "common.hpp"
#include "util/math_functions.hpp"

#include "plugin/concatPlugin.hpp"
//#include "plugin/concatNewPlugin.hpp"
#include "plugin/priorboxPlugin.hpp"
#include "plugin/detectionOutputPlugin.hpp"
#include "plugin/softmaxPlugin.hpp"
#include "plugin/printPlugin.hpp"
//#include "plugin/convPlugin.hpp"
#include "plugin/normPlugin.hpp"


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

/*#define CHECK(status)												\
    {																\
	if (status != 0)												\
	{																\
	    std::cout << "Cuda failure: " << status;					\
	    abort();													\
	}																\
    }*/


// stuff we know about the network and the caffe input/output blobs
static const int INPUT_C = 3;
static const int INPUT_H = 300;//375;
static const int INPUT_W = 300;//500;
static const int IM_INFO_SIZE = 3;
static const int OUTPUT_CLS_SIZE = 21;
static const int OUTPUT_BBOX_SIZE = OUTPUT_CLS_SIZE * 4;

const std::string CLASSES[OUTPUT_CLS_SIZE]{ "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };

const char* INPUT_BLOB_NAME0 = "data";
const char* INPUT_BLOB_NAME1 = "im_info";
const char* OUTPUT_BLOB_NAME0 = "bbox_pred";
const char* OUTPUT_BLOB_NAME1 = "cls_prob";
const char* OUTPUT_BLOB_NAME2 = "rois";
const char* OUTPUT_BLOB_NAME3 = "count";


const int poolingH = 7;
const int poolingW = 7;
const int featureStride = 16;
const int preNmsTop = 6000;
const int nmsMaxOut = 300;
const int anchorsRatioCount = 3;
const int anchorsScaleCount = 3;
const float iouThreshold = 0.7f;
const float minBoxSize = 16;
const float spatialScale = 0.0625f;
const float anchorsRatios[anchorsRatioCount] = { 0.5f, 1.0f, 2.0f };
const float anchorsScales[anchorsScaleCount] = { 8.0f, 16.0f, 32.0f };


//const char* INTPUT_BLOB_NAME4 = "data";
const char* OUTPUT_BLOB_NAME4 = "output";
const char* INPUT_BLOB_NAME2 = "data1";
const char* INPUT_BLOB_NAME3 = "data2";

const int dataDim = 1*3*300*300;
const int dataDim1 = 69952;
const int dataDim2 = 34976*2;
const int outDim = 1400;


// Logger for GIE info/warning/errors
class Logger : public ILogger
{
	void log(Severity severity, const char* msg) override
	{
		// suppress info-level messages
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
} gLogger;

struct PPM
{
	std::string magic, fileName;
	int h, w, max;
	uint8_t buffer[INPUT_C*INPUT_H*INPUT_W];
};

struct BBox
{
	float x1, y1, x2, y2;
};

std::string locateFile(const std::string& input)
{
	std::string file = "data/samples/SSD/" + input;
	struct stat info;
	int i, MAX_DEPTH = 10;
	for (i = 0; i < MAX_DEPTH && stat(file.c_str(), &info); i++)
		file = "../" + file;

    if (i == MAX_DEPTH)
    {
		file = std::string("data/SSD/") + input;
		for (i = 0; i < MAX_DEPTH && stat(file.c_str(), &info); i++)
			file = "../" + file;		
    }

	assert(i != MAX_DEPTH && "Make sure the data is set properly. Check README.txt");

	return file;
}

// simple PPM (portable pixel map) reader
void readPPMFile(const std::string& filename, cv::Mat& ppm)
{
	ppm=cv::imread(locateFile(filename));//载入3通道的彩色图像
	/*ppm.fileName = filename;
	std::ifstream infile(locateFile(filename), std::ifstream::binary);
	infile >> ppm.magic >> ppm.w >> ppm.h >> ppm.max;
	infile.seekg(1, infile.cur);
	infile.read(reinterpret_cast<char*>(ppm.buffer), ppm.w * ppm.h * 3);*/
}

void writePPMFileWithBBox(const std::string& filename, PPM ppm, BBox bbox)
{
	std::ofstream outfile("./" + filename, std::ofstream::binary);
	assert(!outfile.fail());
	outfile << "P6" << "\n" << ppm.w << " " << ppm.h << "\n" << ppm.max << "\n";
	auto round = [](float x)->int {return int(std::floor(x + 0.5f)); };
	for (int x = int(bbox.x1); x < int(bbox.x2); ++x)
	{
		// bbox top border
		ppm.buffer[(round(bbox.y1) * ppm.w + x) * 3] = 255;
		ppm.buffer[(round(bbox.y1) * ppm.w + x) * 3 + 1] = 0;
		ppm.buffer[(round(bbox.y1) * ppm.w + x) * 3 + 2] = 0;
		// bbox bottom border
		ppm.buffer[(round(bbox.y2) * ppm.w + x) * 3] = 255;
		ppm.buffer[(round(bbox.y2) * ppm.w + x) * 3 + 1] = 0;
		ppm.buffer[(round(bbox.y2) * ppm.w + x) * 3 + 2] = 0;
	}
	for (int y = int(bbox.y1); y < int(bbox.y2); ++y)
	{
		// bbox left border
		ppm.buffer[(y * ppm.w + round(bbox.x1)) * 3] = 255;
		ppm.buffer[(y * ppm.w + round(bbox.x1)) * 3 + 1] = 0;
		ppm.buffer[(y * ppm.w + round(bbox.x1)) * 3 + 2] = 0;
		// bbox right border
		ppm.buffer[(y * ppm.w + round(bbox.x2)) * 3] = 255;
		ppm.buffer[(y * ppm.w + round(bbox.x2)) * 3 + 1] = 0;
		ppm.buffer[(y * ppm.w + round(bbox.x2)) * 3 + 2] = 0;
	}
	outfile.write(reinterpret_cast<char*>(ppm.buffer), ppm.w * ppm.h * 3);
}

void caffeToGIEModel(const std::string& deployFile,			// name for caffe prototxt
	const std::string& modelFile,			// name for model 
	const std::vector<std::string>& outputs,		// network outputs
	unsigned int maxBatchSize,				// batch size - NB must be at least as large as the batch we want to run with)
	nvcaffeparser1::IPluginFactory* pluginFactory,	// factory for plugin layers
	IHostMemory **gieModelStream)			// output stream for the GIE model
{
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);

	// parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();
	parser->setPluginFactory(pluginFactory);

	std::cout << "Begin parsing model..." << std::endl;
	const IBlobNameToTensor* blobNameToTensor = parser->parse(locateFile(deployFile).c_str(),
		locateFile(modelFile).c_str(),
		*network,
		DataType::kFLOAT);
	std::cout << "End parsing model..." << std::endl;
	// specify which tensors are outputs
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));


	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(10 << 20);	// we need about 6MB of scratch space for the plugin layer for batch size 5

	std::cout << "Begin building engine..." << std::endl;
	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);
	std::cout << "End building engine..." << std::endl;

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();

	// serialize the engine, then close everything down
	(*gieModelStream) = engine->serialize();

	engine->destroy();
	builder->destroy();
	shutdownProtobufLibrary();
}

void doInference_bak(IExecutionContext& context, float* inputData, float* inputImInfo, float* outputBboxPred, float* outputClsProb, float *outputRois, int batchSize)
{
	const ICudaEngine& engine = context.getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly 2 inputs and 4 outputs.
	assert(engine.getNbBindings() == 6);
	void* buffers[6];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex0 = engine.getBindingIndex(INPUT_BLOB_NAME0),
		inputIndex1 = engine.getBindingIndex(INPUT_BLOB_NAME1),
		outputIndex0 = engine.getBindingIndex(OUTPUT_BLOB_NAME0),
		outputIndex1 = engine.getBindingIndex(OUTPUT_BLOB_NAME1),
		outputIndex2 = engine.getBindingIndex(OUTPUT_BLOB_NAME2),
		outputIndex3 = engine.getBindingIndex(OUTPUT_BLOB_NAME3);


	// create GPU buffers and a stream
	newCHECK(cudaMalloc(&buffers[inputIndex0], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));   // data
	newCHECK(cudaMalloc(&buffers[inputIndex1], batchSize * IM_INFO_SIZE * sizeof(float)));                  // im_info
	newCHECK(cudaMalloc(&buffers[outputIndex0], batchSize * nmsMaxOut * OUTPUT_BBOX_SIZE * sizeof(float))); // bbox_pred
	newCHECK(cudaMalloc(&buffers[outputIndex1], batchSize * nmsMaxOut * OUTPUT_CLS_SIZE * sizeof(float)));  // cls_prob
	newCHECK(cudaMalloc(&buffers[outputIndex2], batchSize * nmsMaxOut * 4 * sizeof(float)));                // rois
	newCHECK(cudaMalloc(&buffers[outputIndex3], batchSize * sizeof(float)));                                // count

	cudaStream_t stream;
	newCHECK(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	newCHECK(cudaMemcpyAsync(buffers[inputIndex0], inputData, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyDefault, stream));
	newCHECK(cudaMemcpyAsync(buffers[inputIndex1], inputImInfo, batchSize * IM_INFO_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueue(batchSize, buffers, stream, nullptr);
	newCHECK(cudaMemcpyAsync(outputBboxPred, buffers[outputIndex0], batchSize * nmsMaxOut * OUTPUT_BBOX_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	newCHECK(cudaMemcpyAsync(outputClsProb, buffers[outputIndex1], batchSize * nmsMaxOut * OUTPUT_CLS_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	newCHECK(cudaMemcpyAsync(outputRois, buffers[outputIndex2], batchSize * nmsMaxOut * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);


	// release the stream and the buffers
	cudaStreamDestroy(stream);
	newCHECK(cudaFree(buffers[inputIndex0]));
	newCHECK(cudaFree(buffers[inputIndex1]));
	newCHECK(cudaFree(buffers[outputIndex0]));
	newCHECK(cudaFree(buffers[outputIndex1]));
	newCHECK(cudaFree(buffers[outputIndex2]));
	newCHECK(cudaFree(buffers[outputIndex3]));
}

void doInference(IExecutionContext& context, float* inputData, float* outputData, int batchSize)
{
	const ICudaEngine& engine = context.getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly 2 inputs and 4 outputs.
	assert(engine.getNbBindings() == 2);
	void* buffers[4];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex0 = engine.getBindingIndex(INPUT_BLOB_NAME0),
		outputIndex0 = engine.getBindingIndex(OUTPUT_BLOB_NAME4);



	// create GPU buffers and a stream
	newCHECK(cudaMalloc(&buffers[inputIndex0], batchSize * dataDim * sizeof(float)));   // data
	//newCHECK(cudaMalloc(&buffers[inputIndex1], batchSize * dataDim1 * sizeof(float)));   // data1
	//newCHECK(cudaMalloc(&buffers[inputIndex2], batchSize * dataDim2 * sizeof(float)));   // data2
	newCHECK(cudaMalloc(&buffers[outputIndex0], batchSize * outDim * sizeof(float)));   // reshapedata

	cudaStream_t stream;
	newCHECK(cudaStreamCreate(&stream));
	static clock_t start_clock_t =  clock();;
	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	newCHECK(cudaMemcpyAsync(buffers[inputIndex0], inputData, batchSize * dataDim * sizeof(float), cudaMemcpyHostToDevice, stream));
	//newCHECK(cudaMemcpyAsync(buffers[inputIndex1], inputData1, batchSize * dataDim1 * sizeof(float), cudaMemcpyHostToDevice, stream));
	//newCHECK(cudaMemcpyAsync(buffers[inputIndex2], inputData2, batchSize * dataDim2 * sizeof(float), cudaMemcpyHostToDevice, stream));
	// newCHECK(cudaMemcpyAsync(buffers[inputIndex1], inputImInfo, batchSize * IM_INFO_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
	std::cout << "begin enqueue..." << std::endl;
	std::cout <<  clock() << std::endl;
	context.enqueue(batchSize, buffers, stream, nullptr);
	std::cout << "end enqueue..." << std::endl;
	// newCHECK(cudaMemcpyAsync(outputBboxPred, buffers[outputIndex0], batchSize * nmsMaxOut * OUTPUT_BBOX_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	// newCHECK(cudaMemcpyAsync(outputClsProb, buffers[outputIndex1], batchSize * nmsMaxOut * OUTPUT_CLS_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	newCHECK(cudaMemcpyAsync(outputData, buffers[outputIndex0], batchSize * outDim * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);
	std::cout <<  CLOCKS_PER_SEC << std::endl;
	std::cout <<  static_cast<double>( clock() - start_clock_t ) / CLOCKS_PER_SEC << std::endl;



	// release the stream and the buffers
	cudaStreamDestroy(stream);
	newCHECK(cudaFree(buffers[outputIndex0]));
	// newCHECK(cudaFree(buffers[inputIndex1]));
	// newCHECK(cudaFree(buffers[outputIndex0]));
	// newCHECK(cudaFree(buffers[outputIndex1]));
	// newCHECK(cudaFree(buffers[outputIndex2]));
	// newCHECK(cudaFree(buffers[outputIndex3]));
}

class Flatten : public IPlugin
{
public:
	Flatten() {}
	Flatten(const void* buffer, size_t size)
	{
		std::cout << size << std::endl;
		assert(size == sizeof(mCopySize));
		mCopySize = *reinterpret_cast<const size_t*>(buffer);
	}

	int getNbOutputs() const override
	{
		//std::cout << "getNbOutputs" << std::endl;
		return 1;
	}
	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
	{
		std::cout << "flatten getOutputDimensions" << std::endl;
		assert(nbInputDims == 1);
		assert(index == 0 && inputs[0].nbDims == 3);

		/*Dims dim;
		dim.nbDims = 1;
		dim.d[0] = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];
		dim.type[0] = DimensionType::kSPATIAL;*/
		DimsCHW chw = DimsCHW(inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2], 1, 1);
		std::cout << inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2] << std::endl;
		//chw.nbDims = 3;
		return chw;
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
		/*std::cout << "flatten enqueue" << std::endl;
		float* data = new float[100];
		newCHECK(cudaMemcpyAsync(data, inputs[0], 100 * sizeof(float), cudaMemcpyDeviceToHost, stream));
		for(int i=0; i < 100; i++) std::cout << data[i] << " ";
		std::cout << std::endl;*/
		std::cout << "Flatten enqueue" << std::endl;
		clock_t start_clock = clock();
		std::cout <<  clock() << std::endl;
		newCHECK(cudaMemcpyAsync(outputs[0], inputs[0], mCopySize * batchSize, cudaMemcpyDeviceToDevice, stream));
  	    static clock_t clock_sum = 0;
	    clock_sum += (clock() - start_clock);
	    std::cout <<  clock() << std::endl;
	    std::cout <<  static_cast<double>( clock_sum ) / CLOCKS_PER_SEC << std::endl;
		return 0;
	}

	size_t getSerializationSize() override
	{
		std::cout << sizeof(mCopySize) << std::endl;
		return sizeof(mCopySize);
	}

	void serialize(void* buffer) override
	{
		*reinterpret_cast<size_t*>(buffer) = mCopySize;
	}

	void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int)	override
	{
		mCopySize = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2] * sizeof(float);
	}

protected:
	size_t mCopySize;
};


template<int OutC>
class Reshape : public IPlugin
{
public:
	Reshape() {}
	Reshape(const void* buffer, size_t size)
	{
		assert(size == sizeof(mCopySize));
		mCopySize = *reinterpret_cast<const size_t*>(buffer);
	}

	int getNbOutputs() const override
	{
		return 1;
	}
	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
	{
		std::cout << "reshape getOutputDimensions" << std::endl;
		assert(nbInputDims == 1);
		assert(index == 0);
		assert(inputs[index].nbDims == 3);
		std::cout << inputs[0].d[0] << std::endl;
		std::cout << inputs[0].d[1] << std::endl;
		std::cout << inputs[0].d[2] << std::endl;
		assert((inputs[0].d[0]*inputs[0].d[1]*inputs[0].d[2]) % OutC == 0);
		/*Dims dims;
		dims.nbDims = 2;
		dims.d[0] = inputs[0].d[1] * inputs[0].d[2] / OutC;
		dims.d[1] = OutC;
		dims.type[0]=dims.type[1]=DimensionType::kSPATIAL;
		return dims;*/
		return DimsCHW(inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2] / OutC, OutC, 1);
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
		/*float *data = new float[mCopySize * batchSize];
		std::cout << "Reshape enqueue" << std::endl;
		newCHECK(cudaMemcpyAsync(data, inputs[0], mCopySize * batchSize, cudaMemcpyDeviceToHost, stream));
		for(int i=0; i < 100; i++) std::cout << data[i] << " ";
		std::cout << std::endl;*/
		std::cout << "Reshape enqueue" << std::endl;
		clock_t start_clock = clock();
		std::cout <<  clock() << std::endl;
		newCHECK(cudaMemcpyAsync(outputs[0], inputs[0], mCopySize * batchSize, cudaMemcpyDeviceToDevice, stream));
  	    static clock_t clock_sum = 0;
	    clock_sum += (clock() - start_clock);
	    std::cout <<  clock() << std::endl;
	    std::cout <<  static_cast<double>( clock_sum ) / CLOCKS_PER_SEC << std::endl;
		return 0;
	}

	size_t getSerializationSize() override
	{
		return sizeof(mCopySize);
	}

	void serialize(void* buffer) override
	{
		*reinterpret_cast<size_t*>(buffer) = mCopySize;
	}

	void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int)	override
	{
		mCopySize = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2] * sizeof(float);
	}

protected:
	size_t mCopySize;
};

class PermuteWHC : public IPlugin
{
public:
	PermuteWHC() {}
	PermuteWHC(const void* data, size_t size)
	{
		const char* d = reinterpret_cast<const char*>(data), *a = d;
		for(int i = 0; i < 3; i++)
		{
			mInputShape[i] = read<int>(d);
		}
		for(int i = 0; i < 4; i++)
		{
			mOld_steps[i] = read<int>(d);
		}
		for(int i = 0; i < 4; i++)
		{
			mNew_steps[i] = read<int>(d);
		}

		assert(d == a + size);
	}

	int getNbOutputs() const override
	{
		return 1;
	}
	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
	{
		std::cout << "permute getOutputDimensions" << std::endl;
		assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
		mInputShape[0] = inputs[0].d[0];
		mInputShape[1] = inputs[0].d[1];
		mInputShape[2] = inputs[0].d[2];
		std::cout << "mInputShape[0] " << mInputShape[0] << std::endl;
		std::cout << "mInputShape[1] " << mInputShape[1] << std::endl;
		std::cout << "mInputShape[2] " << mInputShape[2] << std::endl;
		int permute_order[3] = {1,2,0};
		mOld_steps[3]=1;
		for(int i = 2; i >=0; i--)
		{
			mOld_steps[i] = mOld_steps[i+1]*mInputShape[i];
		}
		mNew_steps[3]=1;
		for(int i = 2; i >=0; i--)
		{
			int order = permute_order[i];
			mNew_steps[i] = mNew_steps[i+1]*mInputShape[order];
		}
		return DimsCHW(inputs[0].d[1], inputs[0].d[2], inputs[0].d[0]);
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
	
	void Permute_(const int count, const float* bottom_data, const bool forward,
		const int* permute_order, const int* old_steps, const int* new_steps,
		const int num_axes, float* top_data) {
		//std::cout << "begin Permute_" << std::endl;
		/*for(int i = 0; i < 3; i++)
		{
			std::cout << mInputShape[i] << std::endl;
		}
		for(int i = 0; i < 4; i++)
		{
			std::cout << mOld_steps[i] <<  std::endl;
		}
		for(int i = 0; i < 4; i++)
		{
			std::cout << mNew_steps[i] <<  std::endl;
		}*/
		//std::cout << count << forward << std::endl;
		for (int i = 0; i < count; ++i) {
		  int old_idx = 0;
		  int idx = i;
		  for (int j = 0; j < num_axes; ++j) {
			int order = permute_order[j];
			old_idx += (idx / new_steps[j]) * old_steps[order];
			//std::cout << j << idx / new_steps[j] << std::endl;
			//std::cout << old_idx << std::endl;
			idx %= new_steps[j];
			//std::cout << i << idx << std::endl;
		  }
		  if (forward) {
			//std::cout << "124" << std::endl;
			top_data[i] = bottom_data[old_idx];
		  } else {
			//bottom_data[old_idx] = top_data[i];
		  }
		}
	}

	// currently it is not possible for a plugin to execute "in place". Therefore we memcpy the data from the input to the output buffer
	int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override
	{
		//const float* bottom_data = reinterpret_cast<const float*>(inputs[0]);
		//float* top_data = reinterpret_cast<float*>(outputs[0]);
		std::cout << "PermuteWHC enqueue" << std::endl;
		clock_t start_clock = clock();
		std::cout <<  clock() << std::endl;
		const int data_dim = mInputShape[0]*mInputShape[1]*mInputShape[2];
		float *bottom_data = new float[data_dim];
		float *top_data = new float[data_dim];
		const int permute_order[4] = {0,2,3,1};
		int num_axes_ = 4;
		
		bool forward = true;
		newCHECK(cudaMemcpyAsync(bottom_data, inputs[0], batchSize * data_dim * sizeof(float), cudaMemcpyDeviceToHost, stream));

		Permute_(data_dim, bottom_data, forward, permute_order, mOld_steps,
				mNew_steps, num_axes_, top_data);
		newCHECK(cudaMemcpyAsync(outputs[0], top_data, batchSize * data_dim * sizeof(float), cudaMemcpyHostToDevice, stream));
  	    static clock_t clock_sum = 0;
	    clock_sum += (clock() - start_clock);
	    std::cout <<  clock() << std::endl;
	    std::cout <<  static_cast<double>( clock_sum ) / CLOCKS_PER_SEC << std::endl;
		delete[] bottom_data;
		delete[] top_data;
		return 0;
	}

	size_t getSerializationSize() override
	{
		return sizeof(int)*11;
	}

	void serialize(void* buffer) override
	{
		char* d = reinterpret_cast<char*>(buffer), *a = d;
		for(int i = 0; i < 3; i++)
		{
			write(d, mInputShape[i]);
		}
		for(int i = 0; i < 4; i++)
		{
			write(d, mOld_steps[i]);
		}
		for(int i = 0; i < 4; i++)
		{
			write(d, mNew_steps[i]);
		}

		assert(d == a + getSerializationSize());
	}

	void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int)	override
	{
		//mCopySize = inputs[0].d[0] * inputs[0].d[1] * sizeof(float);
	}

protected:
	//size_t mCopySize;
	int mInputShape[3], mOld_steps[4], mNew_steps[4];
	
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




// integration for serialization
class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory
{
public:
	// deserialization plugin implementation
	virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override
	{
		std::cout << "layerName " << layerName << std::endl;
		//std::cout << "flatten " << strstr(layerName, "_flat") << std::endl;
		//assert(isPlugin(layerName));
		/*if (!strcmp(layerName, "ReshapeWTo8"))
		{
			assert(mPluginRshp8 == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			mPluginRshp8 = std::unique_ptr<Reshape<8>>(new Reshape<8>());
			return mPluginRshp8.get();
		}
		else if (!strcmp(layerName, "norm"))
		{
			assert(mPluginNorm == nullptr);
			assert(nbWeights == 1 && weights[0].type == DataType::kFLOAT);
			mPluginNorm = std::unique_ptr<NormPlugin>(new NormPlugin(weights, nbWeights));
			return mPluginNorm.get();
		}
		else if (!strcmp(layerName, "RPROIFused"))
		{
			assert(mPluginRPROI == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			mPluginRPROI = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
				(createFasterRCNNPlugin(featureStride, preNmsTop, nmsMaxOut, iouThreshold, minBoxSize, spatialScale,
					DimsHW(poolingH, poolingW), Weights{ nvinfer1::DataType::kFLOAT, anchorsRatios, anchorsRatioCount },
					Weights{ nvinfer1::DataType::kFLOAT, anchorsScales, anchorsScaleCount }), nvPluginDeleter);
			return mPluginRPROI.get();
		}*/
		if (strstr(layerName, "_flat") != nullptr)
		{
			/*assert(mPluginFlatten == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			mPluginFlatten = std::unique_ptr<Flatten>(new Flatten());
			return mPluginFlatten.get();*/
			assert(mapFlattenPlugin.find(layerName) == mapFlattenPlugin.end());
			assert(nbWeights == 0 && weights == nullptr);
			mapFlattenPlugin.insert(std::pair<string, std::unique_ptr<Flatten>>(layerName, std::unique_ptr<Flatten>(new Flatten())));
			std::cout << "123" << std::endl;
			return mapFlattenPlugin[layerName].get();
		}
		/*else if (!strcmp(layerName, "fc6"))
		{
			assert(mPluginConv == nullptr);
			assert(nbWeights == 2);
			mPluginConv = std::unique_ptr<Convolution>(new Convolution(weights, nbWeights));
			return mPluginConv.get();
		}*/
		else if (!strcmp(layerName, "printDims"))
		{
			std::cout << "print" << std::endl;
			assert(mPluginPrint == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			mPluginPrint = std::unique_ptr<PrintDims>(new PrintDims());
			return mPluginPrint.get();
		}
		/*else if (!strcmp(strchr(layerName, '_'), "_norm"))
		{
			assert(mapNormPlugin.find(layerName) == mapNormPlugin.end());
			assert(nbWeights == 1 && weights[0].type == DataType::kFLOAT);
			mapNormPlugin.insert(std::pair<string, std::unique_ptr<NormPlugin>>(layerName, std::unique_ptr<NormPlugin>(new NormPlugin(weights, nbWeights))));
			return mapNormPlugin[layerName].get();
		}*/
		else if (!strcmp(layerName, "mbox_conf_reshape"))
		{
			assert(mPluginRshp8 == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			mPluginRshp8 = std::unique_ptr<Reshape<21>>(new Reshape<21>());
			return mPluginRshp8.get();
		}
		else if (!strncmp(strchr(layerName, '_'), "_norm_mbox_", 11) || (strstr(layerName, "mbox_loc_perm") != nullptr || strstr(layerName, "mbox_conf_perm") != nullptr))
		{
			assert(mapPermutePlugin.find(layerName) == mapPermutePlugin.end());
			//assert(mPluginPermuWHC == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			//mPluginPermuWHC = std::unique_ptr<PermuteWHC>(new PermuteWHC());
			mapPermutePlugin.insert(std::pair<string, std::unique_ptr<PermuteWHC>>(layerName, std::unique_ptr<PermuteWHC>(new PermuteWHC())));
			return mapPermutePlugin[layerName].get();
		}
		/*else if (!strcmp(layerName, "mbox_loc") || !strcmp(layerName, "mbox_conf"))
		{
			//assert(mapConcatPlugin.find(layerName) == mapConcatPlugin.end());

			assert(mapConcatPlugin.find(layerName) == mapConcatPlugin.end());
			assert(nbWeights == 0 && weights == nullptr);
	
			mapConcatPlugin.insert(std::pair<string, std::unique_ptr<ConcatNew>>(layerName, std::unique_ptr<ConcatNew>(new ConcatNew())));
			std::cout << "mapConcatPlugin" << std::endl;
			return mapConcatPlugin[layerName].get();
		}*/
		else if (!strcmp(strchr(layerName, '_'), "_norm_mbox_priorbox") || strstr(layerName, "_mbox_priorbox") != nullptr) 
		{
			/*assert(mPluginPriorBox1 == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			mPluginPriorBox1 = std::unique_ptr<PriorBox<1>>(new PriorBox<1>());
			return mPluginPriorBox1.get();*/
			assert(mapPriorBoxPlugin.find(layerName) == mapPriorBoxPlugin.end());
			assert(nbWeights == 0 && weights == nullptr);
			static int cnt = 1;
			mapPriorBoxPlugin.insert(std::pair<string, std::unique_ptr<PriorBox>>(layerName, std::unique_ptr<PriorBox>(new PriorBox(cnt++))));
			std::cout << "get" <<std::endl;
			return mapPriorBoxPlugin[layerName].get();
			std::cout << cnt <<std::endl;
		}
		else if (!strcmp(layerName, "mbox_priorbox"))
		{
			assert(mPluginConcat == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			mPluginConcat = std::unique_ptr<Concat<1>>(new Concat<1>());
			return mPluginConcat.get();
		}
		else if (!strcmp(layerName, "detection_out"))
		{
			assert(mPluginDetectionOut == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			mPluginDetectionOut = std::unique_ptr<DetectionOutput<21>>(new DetectionOutput<21>());
			return mPluginDetectionOut.get();
		}
		else if (!strcmp(layerName, "conv4_3_norm"))
		{
			assert(mPluginNorm == nullptr);
			assert(nbWeights == 1 && weights[0].type == DataType::kFLOAT);
			mPluginNorm = std::unique_ptr<NormPlugin>(new NormPlugin(weights, nbWeights));
			return mPluginNorm.get();
		}
		else if (!strcmp(layerName, "mbox_conf_softmax"))
		{
			assert(mPluginSoftmax == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			mPluginSoftmax = std::unique_ptr<Softmax<1>>(new Softmax<1>());
			return mPluginSoftmax.get();
		}
		else
		{
			assert(0);
			return nullptr;
		}
	}

	IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
	{
		//assert(isPlugin(layerName));
		/*if (!strcmp(layerName, "ReshapeWTo8"))
		{
			assert(mPluginRshp8 == nullptr);
			mPluginRshp8 = std::unique_ptr<Reshape<8>>(new Reshape<8>(serialData, serialLength));
			return mPluginRshp8.get();
		}
		else if (!strcmp(layerName, "norm"))
		{
			assert(mPluginNorm == nullptr);
			mPluginNorm = std::unique_ptr<NormPlugin>(new NormPlugin(serialData, serialLength));
			return mPluginNorm.get();
		}
		else if (!strcmp(layerName, "RPROIFused"))
		{
			assert(mPluginRPROI == nullptr);
			mPluginRPROI = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createFasterRCNNPlugin(serialData, serialLength), nvPluginDeleter);
			return mPluginRPROI.get();
		}*/
		if (strstr(layerName, "_flat") != nullptr)
		{
			/*assert(mPluginFlatten == nullptr);
			mPluginFlatten = std::unique_ptr<Flatten>(new Flatten(serialData, serialLength));
			return mPluginFlatten.get();*/
			assert(mapFlattenPlugin.find(layerName) != mapFlattenPlugin.end() && mapFlattenPlugin[layerName] == nullptr);
			mapFlattenPlugin[layerName] = std::unique_ptr<Flatten>(new Flatten(serialData, serialLength));
			return mapFlattenPlugin[layerName].get();
		}
		/*else if (!strcmp(layerName, "fc6"))
		{
			assert(mPluginConv == nullptr);
			mPluginConv = std::unique_ptr<Convolution>(new Convolution(serialData, serialLength));
			return mPluginConv.get();
		}*/
		/*if (!strcmp(strchr(layerName, '_'), "_norm"))
		{
			assert(mapNormPlugin.find(layerName) != mapNormPlugin.end() && mapNormPlugin[layerName] == nullptr);
			//std::unique_ptr<NormPlugin> temPlugin = mapNormPlugin[layerName];
			mapNormPlugin[layerName] = std::unique_ptr<NormPlugin>(new NormPlugin(serialData, serialLength));
			//mapNormPlugin[layerName] = temPlugin;
			return mapNormPlugin[layerName].get();
		}*/
		else if (!strcmp(layerName, "mbox_conf_reshape"))
		{
			assert(mPluginRshp8 == nullptr);
			mPluginRshp8 = std::unique_ptr<Reshape<21>>(new Reshape<21>(serialData, serialLength));
			return mPluginRshp8.get();
		}
		else if (!strncmp(strchr(layerName, '_'), "_norm_mbox_", 11) || (strstr(layerName, "mbox_loc_perm") != nullptr || strstr(layerName, "mbox_conf_perm") != nullptr))
		{
			assert(mapPermutePlugin.find(layerName) != mapPermutePlugin.end() && mapPermutePlugin[layerName] == nullptr);
			//assert(mPluginPermuWHC == nullptr);
			mapPermutePlugin[layerName] = std::unique_ptr<PermuteWHC>(new PermuteWHC(serialData, serialLength));
			//mPluginPermuWHC = std::unique_ptr<PermuteWHC>(new PermuteWHC(serialData, serialLength));
			return mapPermutePlugin[layerName].get();
		}
		/*else if (!strcmp(layerName, "mbox_loc") || !strcmp(layerName, "mbox_conf"))
		{
			assert(mapConcatPlugin.find(layerName) != mapConcatPlugin.end() && mapConcatPlugin[layerName] == nullptr);
			mapConcatPlugin[layerName] = std::unique_ptr<ConcatNew>(new ConcatNew(serialData, serialLength));
			return mapConcatPlugin[layerName].get();
		}*/
		else if (!strcmp(strchr(layerName, '_'), "_norm_mbox_priorbox") || strstr(layerName, "_mbox_priorbox") != nullptr)
		{
			/*assert(mPluginPriorBox1 == nullptr);
			mPluginPriorBox1 = std::unique_ptr<PriorBox<1>>(new PriorBox<1>(serialData, serialLength));
			return mPluginPriorBox1.get();*/
			
			assert(mapPriorBoxPlugin.find(layerName) != mapPriorBoxPlugin.end() && mapPriorBoxPlugin[layerName] == nullptr);
			mapPriorBoxPlugin[layerName] = std::unique_ptr<PriorBox>(new PriorBox(serialData, serialLength));
			return mapPriorBoxPlugin[layerName].get();
		}
		else if (!strcmp(layerName, "mbox_priorbox"))
		{
			assert(mPluginConcat == nullptr);
			mPluginConcat = std::unique_ptr<Concat<1>>(new Concat<1>(serialData, serialLength));
			return mPluginConcat.get();
		}
		else if (!strcmp(layerName, "detection_out"))
		{
			assert(mPluginDetectionOut == nullptr);
			mPluginDetectionOut = std::unique_ptr<DetectionOutput<21>>(new DetectionOutput<21>(serialData, serialLength));
			return mPluginDetectionOut.get();
		}
		else if (!strcmp(layerName, "conv4_3_norm"))
		{
			assert(mPluginNorm == nullptr);
			mPluginNorm = std::unique_ptr<NormPlugin>(new NormPlugin(serialData, serialLength));
			return mPluginNorm.get();
		}
		else if (!strcmp(layerName, "mbox_conf_softmax"))
		{
			assert(mPluginSoftmax == nullptr);
			mPluginSoftmax = std::unique_ptr<Softmax<1>>(new Softmax<1>(serialData, serialLength));
			return mPluginSoftmax.get();
		}
		else if (!strcmp(layerName, "printDims"))
		{
			assert(mPluginPrint == nullptr);
			mPluginPrint = std::unique_ptr<PrintDims>(new PrintDims(serialData, serialLength));
			return mPluginPrint.get();
		}
		else
		{
			assert(0);
			return nullptr;
		}
	}

	//caffe parser plugin implementation
	bool isPlugin(const char* name) override
	{
		//return false;
		//std::cout << name << std::endl;
		//if(!strcmp(name, "conv4_3_norm_mbox_loc_flat")) return false;
		//const char* tem = strchr(name, '_');
		//char* tem = "123";
		//if (!tem) return false;conv4_3_norm_mbox_priorbox
		return (strstr(name, "mbox_loc_perm") != nullptr || strstr(name, "mbox_conf_perm") != nullptr)
			|| (strstr(name, "_mbox_priorbox") != nullptr)
			|| (strstr(name, "_flat") != nullptr)
			//|| !strncmp(tem, "_norm_mbox_", 11)
			//|| !strcmp(name, "First_norm")
			//|| !strcmp(name, "fc6")
			//|| !strcmp(tem, "_norm")
			|| !strcmp(name, "Flatten")
			//|| !strcmp(name, "mbox_loc")
			//|| !strcmp(name, "mbox_conf")
			//|| !strcmp(tem, "_norm_mbox_priorbox")
			|| !strcmp(name, "mbox_priorbox") //concat 2
			|| !strcmp(name, "detection_out")
			|| !strcmp(name, "mbox_conf_softmax")
			|| !strcmp(name, "conv4_3_norm")
			|| !strcmp(name, "printDims")
			|| !strcmp(name, "mbox_conf_reshape");
	}

	// the application has to destroy the plugin when it knows it's safe to do so
	void destroyPlugin()
	{
		mPluginRshp8.release();			mPluginRshp8 = nullptr;
		mPluginNorm.release();			mPluginNorm = nullptr;
		mPluginPermuWHC.release();		mPluginPermuWHC = nullptr;
		mPluginFlatten.release();		mPluginFlatten = nullptr;
		mPluginConcat.release();		mPluginConcat = nullptr;
		//mPluginPriorBox1.release();		mPluginPriorBox1 = nullptr;
		mPluginDetectionOut.release();	mPluginDetectionOut = nullptr;
		mPluginSoftmax.release();	mPluginSoftmax = nullptr;
		mPluginPrint.release();	mPluginPrint = nullptr;
		//mPluginConv.release();	mPluginConv = nullptr;
		
		//std::map<string, std::unique_ptr<NormPlugin>>::iterator it = mapNormPlugin.begin();
		for(auto it = mapNormPlugin.begin(); it != mapNormPlugin.end(); it++)
		{
			it->second.release();
			it->second = nullptr;
		}

		//std::map<string, std::unique_ptr<PermuteWHC>>::iterator it = mapPermutePlugin.begin();
		for(auto it = mapPermutePlugin.begin(); it != mapPermutePlugin.end(); it++)
		{
			it->second.release();
			it->second = nullptr;
		}

		for(auto it = mapPriorBoxPlugin.begin(); it != mapPriorBoxPlugin.end(); it++)
		{
			it->second.release();
			it->second = nullptr;
		}
		
		/*for(auto it = mapConcatPlugin.begin(); it != mapConcatPlugin.end(); it++)
		{
			it->second.release();
			it->second = nullptr;
		}*/

		for(auto it = mapFlattenPlugin.begin(); it != mapFlattenPlugin.end(); it++)
		{
			it->second.release();
			it->second = nullptr;
		}
	}


	std::unique_ptr<Reshape<21>> mPluginRshp8{ nullptr };
	std::unique_ptr<NormPlugin> mPluginNorm{ nullptr };
	std::unique_ptr<PermuteWHC> mPluginPermuWHC{ nullptr };
	std::unique_ptr<Flatten> mPluginFlatten{ nullptr };
	std::unique_ptr<Concat<1>> mPluginConcat{ nullptr };
	//std::unique_ptr<PriorBox<1>> mPluginPriorBox1{ nullptr };
	std::unique_ptr<DetectionOutput<21>> mPluginDetectionOut{ nullptr };
	std::unique_ptr<Softmax<1>> mPluginSoftmax{ nullptr };
	std::unique_ptr<PrintDims> mPluginPrint{ nullptr };
	//std::unique_ptr<Convolution> mPluginConv{ nullptr };

	std::map<string, std::unique_ptr<NormPlugin>> mapNormPlugin;
	std::map<string, std::unique_ptr<PermuteWHC>> mapPermutePlugin;
	std::map<string, std::unique_ptr<PriorBox>> mapPriorBoxPlugin;
	//std::map<string, std::unique_ptr<ConcatNew>> mapConcatPlugin;
	std::map<string, std::unique_ptr<Flatten>> mapFlattenPlugin;

	void(*nvPluginDeleter)(INvPlugin*) { [](INvPlugin* ptr) {ptr->destroy(); } };
	std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mPluginRPROI{ nullptr, nvPluginDeleter };
};


void bboxTransformInvAndClip(float* rois, float* deltas, float* predBBoxes, float* imInfo,
	const int N, const int nmsMaxOut, const int numCls)
{
	float width, height, ctr_x, ctr_y;
	float dx, dy, dw, dh, pred_ctr_x, pred_ctr_y, pred_w, pred_h;
	float *deltas_offset, *predBBoxes_offset, *imInfo_offset;
	for (int i = 0; i < N * nmsMaxOut; ++i)
	{
		width = rois[i * 4 + 2] - rois[i * 4] + 1;
		height = rois[i * 4 + 3] - rois[i * 4 + 1] + 1;
		ctr_x = rois[i * 4] + 0.5f * width;
		ctr_y = rois[i * 4 + 1] + 0.5f * height;
		deltas_offset = deltas + i * numCls * 4;
		predBBoxes_offset = predBBoxes + i * numCls * 4;
		imInfo_offset = imInfo + i / nmsMaxOut * 3;
		for (int j = 0; j < numCls; ++j)
		{
			dx = deltas_offset[j * 4];
			dy = deltas_offset[j * 4 + 1];
			dw = deltas_offset[j * 4 + 2];
			dh = deltas_offset[j * 4 + 3];
			pred_ctr_x = dx * width + ctr_x;
			pred_ctr_y = dy * height + ctr_y;
			pred_w = exp(dw) * width;
			pred_h = exp(dh) * height;
			predBBoxes_offset[j * 4] = std::max(std::min(pred_ctr_x - 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
			predBBoxes_offset[j * 4 + 1] = std::max(std::min(pred_ctr_y - 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
			predBBoxes_offset[j * 4 + 2] = std::max(std::min(pred_ctr_x + 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
			predBBoxes_offset[j * 4 + 3] = std::max(std::min(pred_ctr_y + 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
		}
	}
}

std::vector<int> nms(std::vector<std::pair<float, int> >& score_index, float* bbox, const int classNum, const int numClasses, const float nms_threshold)
{
	auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
		if (x1min > x2min) {
			std::swap(x1min, x2min);
			std::swap(x1max, x2max);
		}
		return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
	};
	auto computeIoU = [&overlap1D](float* bbox1, float* bbox2) -> float {
		float overlapX = overlap1D(bbox1[0], bbox1[2], bbox2[0], bbox2[2]);
		float overlapY = overlap1D(bbox1[1], bbox1[3], bbox2[1], bbox2[3]);
		float area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
		float area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
		float overlap2D = overlapX * overlapY;
		float u = area1 + area2 - overlap2D;
		return u == 0 ? 0 : overlap2D / u;
	};

	std::vector<int> indices;
	for (auto i : score_index)
	{
		const int idx = i.second;
		bool keep = true;
		for (unsigned k = 0; k < indices.size(); ++k)
		{
			if (keep)
			{
				const int kept_idx = indices[k];
				float overlap = computeIoU(&bbox[(idx*numClasses + classNum) * 4],
					&bbox[(kept_idx*numClasses + classNum) * 4]);
				keep = overlap <= nms_threshold;
			}
			else
				break;
		}
		if (keep) indices.push_back(idx);
	}
	return indices;
}

static clock_t start_clock;

void VisualizeBBox_bak(const vector<cv::Mat>& images, const float* detections_data,
                   const float threshold, const vector<cv::Scalar>& colors,
                   const map<int, string>& label_to_display_name,
                   const string& save_file) {
  std::cout << 1 << std::endl;
  // Retrieve detections.
  //CHECK_EQ(detections->width(), 7);
  const int num_det = 200;
  const int num_img = images.size();
  if (num_det == 0 || num_img == 0) {
    return;
  }
  // Comute FPS.
  float fps = num_img / (static_cast<double>(clock() - start_clock) /
          CLOCKS_PER_SEC);

  //const Dtype* detections_data = detections->cpu_data();
  const int width = images[0].cols;
  const int height = images[0].rows;
  vector<LabelBBox> all_detections(num_img);
  for (int i = 0; i < num_det; ++i) {
    const int img_idx = detections_data[i * 7];
    CHECK_LT(img_idx, num_img);
	if(img_idx == -1) break;
    const int label = detections_data[i * 7 + 1];
    const float score = detections_data[i * 7 + 2];
    if (score < threshold) {
      continue;
    }
	//std::cout << label << std::endl;
    NormalizedBBox bbox;
    bbox.set_xmin(detections_data[i * 7 + 3] * width);
    bbox.set_ymin(detections_data[i * 7 + 4] * height);
    bbox.set_xmax(detections_data[i * 7 + 5] * width);
    bbox.set_ymax(detections_data[i * 7 + 6] * height);
    bbox.set_score(score);
    all_detections[img_idx][label].push_back(bbox);
  }
  std::cout << 2 << std::endl;
  int fontface = cv::FONT_HERSHEY_SIMPLEX;
  double scale = 1;
  int thickness = 2;
  int baseline = 0;
  char buffer[50];
  for (int i = 0; i < num_img; ++i) {
    cv::Mat image = images[i];
    // Show FPS.
    snprintf(buffer, sizeof(buffer), "FPS: %.2f", fps);
    cv::Size text = cv::getTextSize(buffer, fontface, scale, thickness,
                                    &baseline);
    cv::rectangle(image, cv::Point(0, 0),
                  cv::Point(text.width, text.height + baseline),
                  CV_RGB(255, 255, 255), CV_FILLED);
    cv::putText(image, buffer, cv::Point(0, text.height + baseline / 2.),
                fontface, scale, CV_RGB(0, 0, 0), thickness, 8);
    // Draw bboxes.
    for (map<int, vector<NormalizedBBox> >::iterator it =
         all_detections[i].begin(); it != all_detections[i].end(); ++it) {
      int label = it->first;
      string label_name = "Unknown";
      if (label_to_display_name.find(label) != label_to_display_name.end()) {
        label_name = label_to_display_name.find(label)->second;
      }
      CHECK_LT(label, colors.size());
      const cv::Scalar& color = colors[label];
      const vector<NormalizedBBox>& bboxes = it->second;
      for (int j = 0; j < bboxes.size(); ++j) {
        cv::Point top_left_pt(bboxes[j].xmin(), bboxes[j].ymin());
        cv::Point bottom_right_pt(bboxes[j].xmax(), bboxes[j].ymax());
        cv::rectangle(image, top_left_pt, bottom_right_pt, color, 4);
        cv::Point bottom_left_pt(bboxes[j].xmin(), bboxes[j].ymax());
        snprintf(buffer, sizeof(buffer), "%s: %.2f", label_name.c_str(),
                 bboxes[j].score());
        cv::Size text = cv::getTextSize(buffer, fontface, scale, thickness,
                                        &baseline);
        cv::rectangle(
            image, bottom_left_pt + cv::Point(0, 0),
            bottom_left_pt + cv::Point(text.width, -text.height-baseline),
            color, CV_FILLED);
        //cv::putText(image, buffer, bottom_left_pt - cv::Point(0, baseline),
        //            fontface, scale, CV_RGB(0, 0, 0), thickness, 8);
      }
    }
    // Save result if required.
    /*if (!save_file.empty()) {
      if (!cap_out.isOpened()) {
        cv::Size size(image.size().width, image.size().height);
        cv::VideoWriter outputVideo(save_file, CV_FOURCC('D', 'I', 'V', 'X'),
            30, size, true);
        cap_out = outputVideo;
      }
      cap_out.write(image);
    }*/
  std::cout << 3 << std::endl;
    cv::imshow("detections", image);
	cv::waitKey(1);
    /*if (cv::waitKey(1) == 27) {
      raise(SIGINT);
    }*/
  }
  start_clock = clock();
}

void VisualizeBBox(const cv::Mat image, const float* detections_data,
                   const float threshold, const vector<cv::Scalar>& colors,
                   const map<int, string>& label_to_display_name,
                   const string& save_file) {
  // Retrieve detections.
  //CHECK_EQ(detections->width(), 7);
  const int num_det = 200;
  // Comute FPS.
  float fps = 1.0 / (static_cast<double>(clock() - start_clock) /
          CLOCKS_PER_SEC);

  //const Dtype* detections_data = detections->cpu_data();
  const int num_img = 1;
  const int width = image.cols;
  const int height = image.rows;
  vector<LabelBBox> all_detections(num_img);
  for (int i = 0; i < num_det; ++i) {
    const int img_idx = detections_data[i * 7];
    CHECK_LT(img_idx, num_img);
	if(img_idx == -1) break;
    const int label = detections_data[i * 7 + 1];
    const float score = detections_data[i * 7 + 2];
    if (score < threshold) {
      continue;
    }
	//std::cout << label << std::endl;
    NormalizedBBox bbox;
    bbox.set_xmin(detections_data[i * 7 + 3] * width);
    bbox.set_ymin(detections_data[i * 7 + 4] * height);
    bbox.set_xmax(detections_data[i * 7 + 5] * width);
    bbox.set_ymax(detections_data[i * 7 + 6] * height);
    bbox.set_score(score);
    all_detections[img_idx][label].push_back(bbox);
  }

  int fontface = cv::FONT_HERSHEY_SIMPLEX;
  double scale = 1;
  int thickness = 2;
  int baseline = 0;
  char buffer[50];
    //cv::Mat image = images[i];
    // Show FPS.
    snprintf(buffer, sizeof(buffer), "FPS: %.2f", fps);
    cv::Size text = cv::getTextSize(buffer, fontface, scale, thickness,
                                    &baseline);
    cv::rectangle(image, cv::Point(0, 0),
                  cv::Point(text.width, text.height + baseline),
                  CV_RGB(255, 255, 255), CV_FILLED);
    cv::putText(image, buffer, cv::Point(0, text.height + baseline / 2.),
                fontface, scale, CV_RGB(0, 0, 0), thickness, 8);
    // Draw bboxes.
    for (map<int, vector<NormalizedBBox> >::iterator it =
         all_detections[0].begin(); it != all_detections[0].end(); ++it) {
      int label = it->first;
      string label_name = "Unknown";
      if (label_to_display_name.find(label) != label_to_display_name.end()) {
        label_name = label_to_display_name.find(label)->second;
      }
      CHECK_LT(label, colors.size());
      const cv::Scalar& color = colors[label];
      const vector<NormalizedBBox>& bboxes = it->second;
      for (int j = 0; j < bboxes.size(); ++j) {
        cv::Point top_left_pt(bboxes[j].xmin(), bboxes[j].ymin());
        cv::Point bottom_right_pt(bboxes[j].xmax(), bboxes[j].ymax());
        cv::rectangle(image, top_left_pt, bottom_right_pt, color, 4);
        cv::Point bottom_left_pt(bboxes[j].xmin(), bboxes[j].ymax());
        snprintf(buffer, sizeof(buffer), "%s: %.2f", label_name.c_str(),
                 bboxes[j].score());
        cv::Size text = cv::getTextSize(buffer, fontface, scale, thickness,
                                        &baseline);
        cv::rectangle(
            image, bottom_left_pt + cv::Point(0, 0),
            bottom_left_pt + cv::Point(text.width, -text.height-baseline),
            color, CV_FILLED);
        cv::putText(image, buffer, bottom_left_pt - cv::Point(0, baseline),
                    fontface, scale, CV_RGB(0, 0, 0), thickness, 8);
      }
    }
    // Save result if required.
    /*if (!save_file.empty()) {
      if (!cap_out.isOpened()) {
        cv::Size size(image.size().width, image.size().height);
        cv::VideoWriter outputVideo(save_file, CV_FOURCC('D', 'I', 'V', 'X'),
            30, size, true);
        cap_out = outputVideo;
      }
      cap_out.write(image);
    }*/
    cv::imshow("detections", image);
	cv::waitKey(1);
    /*if (cv::waitKey(1) == 27) {
      raise(SIGINT);
    }*/
  //start_clock = clock();
}

cv::Scalar HSV2RGB(const float h, const float s, const float v) {
  const int h_i = static_cast<int>(h * 6);
  const float f = h * 6 - h_i;
  const float p = v * (1 - s);
  const float q = v * (1 - f*s);
  const float t = v * (1 - (1 - f) * s);
  float r, g, b;
  switch (h_i) {
    case 0:
      r = v; g = t; b = p;
      break;
    case 1:
      r = q; g = v; b = p;
      break;
    case 2:
      r = p; g = v; b = t;
      break;
    case 3:
      r = p; g = q; b = v;
      break;
    case 4:
      r = t; g = p; b = v;
      break;
    case 5:
      r = v; g = p; b = q;
      break;
    default:
      r = 1; g = 1; b = 1;
      break;
  }
  return cv::Scalar(r * 255, g * 255, b * 255);
}

vector<cv::Scalar> GetColors(const int n) {
  vector<cv::Scalar> colors;
  cv::RNG rng(12345);
  const float golden_ratio_conjugate = 0.618033988749895;
  const float s = 0.3;
  const float v = 0.99;
  for (int i = 0; i < n; ++i) {
    const float h = std::fmod(rng.uniform(0.f, 1.f) + golden_ratio_conjugate,
                              1.f);
    colors.push_back(HSV2RGB(h, s, v));
  }
  return colors;
}

void tranImg(float* &data, cv::Mat img)
{
	float pixelMean[3]{ 63,63,73 }; // also in BGR order
	for (int c = 0; c < INPUT_C; ++c)
	{
		// the color image to input should be in BGR order
		for (unsigned j = 0; j < INPUT_H; ++j)
			for (unsigned k = 0, volChl = INPUT_H*INPUT_W; k < INPUT_W; ++k)
				//data[i*volImg + c*volChl + j] = float(ppms[i].[j*INPUT_C + 2 - c]) - pixelMean[c];
				data[c*volChl + j*INPUT_W + k] = (float)(img.at<cv::Vec3b>(j,k)[c]) - pixelMean[c];
	}
}


int main(int argc, char** argv)
{
	Caffe c;
	// create a GIE model from the caffe model and serialize it to a stream
	PluginFactory pluginFactory;
	IHostMemory *gieModelStream{ nullptr };
	// batch size
	const int N = 1;
	std::cout << "caffeToGIEModel..." << std::endl;
	caffeToGIEModel("IPlugin.prototxt",
		"VGG_VOC0712_SSD_300x300_iter_120000.caffemodel",
		std::vector < std::string > { OUTPUT_BLOB_NAME4 },
		N, &pluginFactory, &gieModelStream);
	std::cout << "caffeToGIEModel done" << std::endl;
	pluginFactory.destroyPlugin();


	// read a random sample image
	//srand(unsigned(time(nullptr)));
	// available images 
	std::vector<std::string> imageList = { "000456.ppm", "004545.ppm" };
	std::vector<cv::Mat> ppms(N);

	//float imInfo[N * 3]; // input im_info	
	/*std::random_shuffle(imageList.begin(), imageList.end(), [](int i) {return rand() % i; });
	assert(ppms.size() <= imageList.size());
	for (int i = 0; i < N; ++i)
	{
		readPPMFile(imageList[1], ppms[i]);
		std::cout << "Reading " << imageList[i] << std::endl;
	}*/


	// deserialize the engine 
	IRuntime* runtime = createInferRuntime(gLogger);
	std::cout << "deserializeCudaEngine..." << std::endl;
	ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), &pluginFactory);
	std::cout << "deserializeCudaEngine done" << std::endl;

	IExecutionContext *context = engine->createExecutionContext();

	// host memory for outputs 
	//float* rois = new float[N * nmsMaxOut * 4];
	//float* bboxPreds = new float[N * nmsMaxOut * OUTPUT_BBOX_SIZE];
	//float* clsProbs = new float[N * nmsMaxOut * OUTPUT_CLS_SIZE];
	//float* data = new float[N*dataDim];
	float* out = new float[outDim];
//layer_width_ * layer_height_ * num_priors_ * 8
	//float kONE = 1.0f;//, kZERO = 0.0f;
	//out[outDim-1] = 23.0f;

	//caffe_set<float>(N*dataDim, kONE, data);
	/*for(int i = 0; i < dataDim; i++)
	{
		data[i] = 0.0f;
	}*/
	/*float* data1 = new float[N*dataDim1];
	for(int i = 0; i < dataDim1; i++)
	{
		data1[i] = 0.8f;
	}
	float* data2 = new float[N*dataDim2];
	for(int i = 0; i < dataDim1; i++)
	{
		data1[i] = 0.1f;
	}
	for(int i = 0; i < dataDim1; i++)
	{
		if(i%4>=2) data1[i] = 0.3f;
	}*/

	// predicted bounding boxes
	//float* predBBoxes = new float[N * nmsMaxOut * OUTPUT_BBOX_SIZE];
	//cv::imshow("img", ppms[0]);
	//cv::waitKey(0);

	//start_clock = clock();
	/*float* data = new float[N*INPUT_C*INPUT_H*INPUT_W];
	// pixel mean used by the Faster R-CNN's author
	//float pixelMean[3]{ 102.9801f, 115.9465f, 122.7717f }; // also in BGR order
	float pixelMean[3]{ 63,63,73 }; // also in BGR order
	//63,63,73
	for (int i = 0, volImg = INPUT_C*INPUT_H*INPUT_W; i < N; ++i)
	{
		cv::Mat img = ppms[0];
		for (int c = 0; c < INPUT_C; ++c)
		{
			// the color image to input should be in BGR order
			for (unsigned j = 0; j < INPUT_H; ++j)
				for (unsigned k = 0, volChl = INPUT_H*INPUT_W; k < INPUT_W; ++k)
					//data[i*volImg + c*volChl + j] = float(ppms[i].[j*INPUT_C + 2 - c]) - pixelMean[c];
					data[i*volImg + c*volChl + j*INPUT_W + k] = (float)(img.at<cv::Vec3b>(j,k)[c]) - pixelMean[c];
		}
	}*/
	/*for(int i = 0; i < 100; i++)
	{
		std::cout << data[i] << ' ';
	}
	std::cout << std::endl;
	for(int i = 0; i < 100; i++)
	{
		std::cout << data[i+90000] << ' ';
	}
	std::cout << std::endl;
	for(int i = 0; i < 100; i++)
	{
		std::cout << data[i+180000] << ' ';
	}
	std::cout << std::endl;*/

	// run inference
	//doInference(*context, data, imInfo, bboxPreds, clsProbs, rois, N);
	/*std::cout << "begin inference..." <<std::endl;
	doInference(*context, data, out, N);
	std::cout << "end inference..." <<std::endl;
	std::cout <<  static_cast<double>( clock() - start_clock ) / CLOCKS_PER_SEC << std::endl;

	std::map<int, string>lableName;
	for(int i = 0; i < 21; i++) lableName[i] = to_string(i);

	VisualizeBBox(ppms, out, 0.5f, GetColors(21), lableName, "");*/
	float* data = new float[N*INPUT_C*INPUT_H*INPUT_W];
	cv::VideoCapture cap(1);
	cap.set(CV_CAP_PROP_FRAME_WIDTH, 300);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 300);
	cap.set(CV_CAP_PROP_FPS ,20);
	cv::Mat img;
	

	start_clock = clock();
	readPPMFile(imageList[1], img);
	//cap >> img;		
	cv::imshow("img", img);
	cv::waitKey(1);
	tranImg(data, img);
	std::cout << "begin inference..." <<std::endl;
	doInference(*context, data, out, N);
	std::cout << "end inference..." <<std::endl;
	std::cout <<  static_cast<double>( clock() - start_clock ) / CLOCKS_PER_SEC << std::endl;
	std::map<int, string>lableName;
	for(int i = 0; i < 21; i++) lableName[i] = to_string(i);
	std::cout << 0 << std::endl;
	VisualizeBBox(img, out, 0.5f, GetColors(21), lableName, "");
	for(int i= 0; i < 7*5/*outDim*/; i++)
	{
		//printf("%f ", out[i]);
		std::cout << out[i] << ' ';
		if ((i+1)%(7)==0)
			printf("\n");
	}
	std::cout <<  "fps::" << 1.0 /(static_cast<double>( clock() - start_clock ) / CLOCKS_PER_SEC) << std::endl;
	cv::waitKey(3000);

	for(int i = 0; i < 10000; i++)
	{
		start_clock = clock();
		//readPPMFile(imageList[i], img);
		cap >> img;		
		if(i%3) continue;
		cv::imshow("img", img);
		cv::waitKey(1);
		tranImg(data, img);
		std::cout << "begin inference..." <<std::endl;
		doInference(*context, data, out, N);
		std::cout << "end inference..." <<std::endl;
		std::cout <<  static_cast<double>( clock() - start_clock ) / CLOCKS_PER_SEC << std::endl;

		std::map<int, string>lableName;
		for(int i = 0; i < 21; i++) lableName[i] = to_string(i);

		VisualizeBBox(img, out, 0.5f, GetColors(21), lableName, "");
		for(int i= 0; i < 7*5/*outDim*/; i++)
		{
			//printf("%f ", out[i]);
			std::cout << out[i] << ' ';
			if ((i+1)%(7)==0)
				printf("\n");
		}
		std::cout <<  "fps::" << 1.0 /(static_cast<double>( clock() - start_clock ) / CLOCKS_PER_SEC) << std::endl;
	}
	/*for(int i= 0; i < outDim; i++)
	{
		//printf("%f ", out[i]);
		std::cout << out[i] << ' ';
		if ((i+1)%(7)==0)
			printf("\n");
	}*/
	/*printf("\n");
	for(int i= 0; i < 100; i++)
	{
		//printf("%f ", out[i]);
		std::cout << out[i+dataDim] << ' ';
		//if (i%(38*38)==8)
			//printf("\n");
	}*/
	// destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();
	pluginFactory.destroyPlugin();

	// unscale back to raw image space
	/* for (int i = 0; i < N; ++i)
	{
		float * rois_offset = rois + i * nmsMaxOut * 4;
		for (int j = 0; j < nmsMaxOut * 4 && imInfo[i * 3 + 2] != 1; ++j)
			rois_offset[j] /= imInfo[i * 3 + 2];
	}

	bboxTransformInvAndClip(rois, bboxPreds, predBBoxes, imInfo, N, nmsMaxOut, OUTPUT_CLS_SIZE);

	const float nms_threshold = 0.3f;
	const float score_threshold = 0.8f;

	for (int i = 0; i < N; ++i)
	{
		float *bbox = predBBoxes + i * nmsMaxOut * OUTPUT_BBOX_SIZE;
		float *scores = clsProbs + i * nmsMaxOut * OUTPUT_CLS_SIZE;
		for (int c = 1; c < OUTPUT_CLS_SIZE; ++c) // skip the background
		{
			std::vector<std::pair<float, int> > score_index;
			for (int r = 0; r < nmsMaxOut; ++r)
			{
				if (scores[r*OUTPUT_CLS_SIZE + c] > score_threshold)
				{
					score_index.push_back(std::make_pair(scores[r*OUTPUT_CLS_SIZE + c], r));
					std::stable_sort(score_index.begin(), score_index.end(),
						[](const std::pair<float, int>& pair1,
							const std::pair<float, int>& pair2) {
						return pair1.first > pair2.first;
					});
				}
			}

			// apply NMS algorithm
			std::vector<int> indices = nms(score_index, bbox, c, OUTPUT_CLS_SIZE, nms_threshold);
			// Show results
			for (unsigned k = 0; k < indices.size(); ++k)
			{
				int idx = indices[k];
				std::string storeName = CLASSES[c] + "-" + std::to_string(scores[idx*OUTPUT_CLS_SIZE + c]) + ".ppm";
				std::cout << "Detected " << CLASSES[c] << " in " << ppms[i].fileName << " with confidence " << scores[idx*OUTPUT_CLS_SIZE + c] * 100.0f << "% "
					<< " (Result stored in " << storeName << ")." << std::endl;

				BBox b{ bbox[idx*OUTPUT_BBOX_SIZE + c * 4], bbox[idx*OUTPUT_BBOX_SIZE + c * 4 + 1], bbox[idx*OUTPUT_BBOX_SIZE + c * 4 + 2], bbox[idx*OUTPUT_BBOX_SIZE + c * 4 + 3] };
				writePPMFileWithBBox(storeName, ppms[i], b);
			}
		}
	} */


	delete[] data;
	//delete[] rois;
	//delete[] bboxPreds;
	//delete[] clsProbs;
	//delete[] predBBoxes;
	return 0;
}


/*int main_old(int argc, char** argv)
{
	// create a GIE model from the caffe model and serialize it to a stream
	PluginFactory pluginFactory;
	IHostMemory *gieModelStream{ nullptr };
	// batch size
	const int N = 2;
	caffeToGIEModel("IPlugin.prototxt",
		"DSOD300_VOC0712_DSOD300_300x300_iter_14000.caffemodel",
		std::vector < std::string > { OUTPUT_BLOB_NAME0, OUTPUT_BLOB_NAME1, OUTPUT_BLOB_NAME2, OUTPUT_BLOB_NAME3 },
		N, &pluginFactory, &gieModelStream);

	pluginFactory.destroyPlugin();

	// read a random sample image
	srand(unsigned(time(nullptr)));
	// available images 
	std::vector<std::string> imageList = { "000456.ppm",  "000542.ppm",  "001150.ppm", "001763.ppm", "004545.ppm" };
	std::vector<PPM> ppms(N);

	float imInfo[N * 3]; // input im_info	
	std::random_shuffle(imageList.begin(), imageList.end(), [](int i) {return rand() % i; });
	assert(ppms.size() <= imageList.size());
	for (int i = 0; i < N; ++i)
	{
		readPPMFile(imageList[i], ppms[i]);
		std::cout << "Reading " << ppms[i].fileName << std::endl;
		imInfo[i * 3] = float(ppms[i].h);   // number of rows
		imInfo[i * 3 + 1] = float(ppms[i].w); // number of columns
		imInfo[i * 3 + 2] = 1;         // image scale
	}

	float* data = new float[N*INPUT_C*INPUT_H*INPUT_W];
	// pixel mean used by the Faster R-CNN's author
	float pixelMean[3]{ 102.9801f, 115.9465f, 122.7717f }; // also in BGR order
	for (int i = 0, volImg = INPUT_C*INPUT_H*INPUT_W; i < N; ++i)
	{
		for (int c = 0; c < INPUT_C; ++c)
		{
			// the color image to input should be in BGR order
			for (unsigned j = 0, volChl = INPUT_H*INPUT_W; j < volChl; ++j)
				data[i*volImg + c*volChl + j] = float(ppms[i].buffer[j*INPUT_C + 2 - c]) - pixelMean[c];
		}
	}

	// deserialize the engine 
	IRuntime* runtime = createInferRuntime(gLogger);
	ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), &pluginFactory);

	IExecutionContext *context = engine->createExecutionContext();


	// host memory for outputs 
	float* rois = new float[N * nmsMaxOut * 4];
	float* bboxPreds = new float[N * nmsMaxOut * OUTPUT_BBOX_SIZE];
	float* clsProbs = new float[N * nmsMaxOut * OUTPUT_CLS_SIZE];

	// predicted bounding boxes
	float* predBBoxes = new float[N * nmsMaxOut * OUTPUT_BBOX_SIZE];

	// run inference
	doInference(*context, data, imInfo, bboxPreds, clsProbs, rois, N);

	// destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();
	pluginFactory.destroyPlugin();

	// unscale back to raw image space
	for (int i = 0; i < N; ++i)
	{
		float * rois_offset = rois + i * nmsMaxOut * 4;
		for (int j = 0; j < nmsMaxOut * 4 && imInfo[i * 3 + 2] != 1; ++j)
			rois_offset[j] /= imInfo[i * 3 + 2];
	}

	bboxTransformInvAndClip(rois, bboxPreds, predBBoxes, imInfo, N, nmsMaxOut, OUTPUT_CLS_SIZE);

	const float nms_threshold = 0.3f;
	const float score_threshold = 0.8f;

	for (int i = 0; i < N; ++i)
	{
		float *bbox = predBBoxes + i * nmsMaxOut * OUTPUT_BBOX_SIZE;
		float *scores = clsProbs + i * nmsMaxOut * OUTPUT_CLS_SIZE;
		for (int c = 1; c < OUTPUT_CLS_SIZE; ++c) // skip the background
		{
			std::vector<std::pair<float, int> > score_index;
			for (int r = 0; r < nmsMaxOut; ++r)
			{
				if (scores[r*OUTPUT_CLS_SIZE + c] > score_threshold)
				{
					score_index.push_back(std::make_pair(scores[r*OUTPUT_CLS_SIZE + c], r));
					std::stable_sort(score_index.begin(), score_index.end(),
						[](const std::pair<float, int>& pair1,
							const std::pair<float, int>& pair2) {
						return pair1.first > pair2.first;
					});
				}
			}

			// apply NMS algorithm
			std::vector<int> indices = nms(score_index, bbox, c, OUTPUT_CLS_SIZE, nms_threshold);
			// Show results
			for (unsigned k = 0; k < indices.size(); ++k)
			{
				int idx = indices[k];
				std::string storeName = CLASSES[c] + "-" + std::to_string(scores[idx*OUTPUT_CLS_SIZE + c]) + ".ppm";
				std::cout << "Detected " << CLASSES[c] << " in " << ppms[i].fileName << " with confidence " << scores[idx*OUTPUT_CLS_SIZE + c] * 100.0f << "% "
					<< " (Result stored in " << storeName << ")." << std::endl;

				BBox b{ bbox[idx*OUTPUT_BBOX_SIZE + c * 4], bbox[idx*OUTPUT_BBOX_SIZE + c * 4 + 1], bbox[idx*OUTPUT_BBOX_SIZE + c * 4 + 2], bbox[idx*OUTPUT_BBOX_SIZE + c * 4 + 3] };
				writePPMFileWithBBox(storeName, ppms[i], b);
			}
		}
	}


	delete[] data;
	delete[] rois;
	delete[] bboxPreds;
	delete[] clsProbs;
	delete[] predBBoxes;
	return 0;
}*/
