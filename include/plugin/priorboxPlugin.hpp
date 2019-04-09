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
#include <vector>
#include <math.h>
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


class PriorBox : public IPlugin
{
public:
	PriorBox(int id) 
	{	
		ID = id;
		/*ID = ID;
		assert(ID >= 1 && ID <= 6);
		//max_sizes_, aspect_ratios_, num_priors_;
		float tem[7] = {30.0f, 60.0f, 111.0f, 162.0f, 213.0f, 264.0f, 315.0f};
		min_sizes_.push_back(tem[ID-1]);
		max_sizes_.push_back(tem[ID]);
		aspect_ratios_.push_back(1.0f);
		aspect_ratios_.push_back(2.0f);
		switch(ID)
		{
			case 1:{
				step_ = 8.0f;
				num_priors_ = aspect_ratios_.size() * min_sizes_.size() + max_sizes_.size();
				break;
			}
			case 2:{
				step_ = 16.0f;
				aspect_ratios_.push_back(3.0f);
				num_priors_ = aspect_ratios_.size() * min_sizes_.size() + max_sizes_.size();
				break;
			}
			case 3:{
				step_ = 32.0f;
				aspect_ratios_.push_back(3.0f);
				num_priors_ = aspect_ratios_.size() * min_sizes_.size() + max_sizes_.size();
				break;
			}
			case 4:{
				step_ = 64.0f;
				aspect_ratios_.push_back(3.0f);
				num_priors_ = aspect_ratios_.size() * min_sizes_.size() + max_sizes_.size();
				break;
			}
			case 5:{
				step_ = 100.0f;
				num_priors_ = aspect_ratios_.size() * min_sizes_.size() + max_sizes_.size();
				break;
			}
			case 6:{
				step_ = 300.0f;
				num_priors_ = aspect_ratios_.size() * min_sizes_.size() + max_sizes_.size();
				break;
			}
			default:{
				assert(0);
			}
		}*/
		initParam();
	}
	
	void initParam()
	{
		assert(ID >= 1 && ID <= 6);
		//max_sizes_, aspect_ratios_, num_priors_;
		float tem[7] = {30.0f, 60.0f, 111.0f, 162.0f, 213.0f, 264.0f, 315.0f};
		min_sizes_.push_back(tem[ID-1]);
		max_sizes_.push_back(tem[ID]);
		aspect_ratios_.push_back(1.0f);
		aspect_ratios_.push_back(2.0f);
		aspect_ratios_.push_back(0.5f);
		switch(ID)
		{
			case 1:{
				step_ = 8.0f;
				num_priors_ = aspect_ratios_.size() * min_sizes_.size() + max_sizes_.size();
				break;
			}
			case 2:{
				step_ = 16.0f;
				aspect_ratios_.push_back(3.0f);
				aspect_ratios_.push_back(1.0f/3.0f);
				num_priors_ = aspect_ratios_.size() * min_sizes_.size() + max_sizes_.size();
				break;
			}
			case 3:{
				step_ = 32.0f;
				aspect_ratios_.push_back(3.0f);
				aspect_ratios_.push_back(1.0f/3.0f);
				num_priors_ = aspect_ratios_.size() * min_sizes_.size() + max_sizes_.size();
				break;
			}
			case 4:{
				step_ = 64.0f;
				aspect_ratios_.push_back(3.0f);
				aspect_ratios_.push_back(1.0f/3.0f);
				num_priors_ = aspect_ratios_.size() * min_sizes_.size() + max_sizes_.size();
				break;
			}
			case 5:{
				step_ = 100.0f;
				num_priors_ = aspect_ratios_.size() * min_sizes_.size() + max_sizes_.size();
				break;
			}
			case 6:{
				step_ = 300.0f;
				num_priors_ = aspect_ratios_.size() * min_sizes_.size() + max_sizes_.size();
				break;
			}
			default:{
				assert(0);
			}
		}
	    std::cout << "num_priors_ " << num_priors_ << std::endl;
	}

	PriorBox(const void* buffer, size_t size)
	{
		const char* d = reinterpret_cast<const char*>(buffer), *a = d;
		
		ID = read<int>(d);
		layer_height_ = read<int>(d);
		layer_width_ = read<int>(d);
		img_height_ = read<int>(d);
		img_width_ = read<int>(d);

		initParam();
		
		assert(d == a + size);
	}

	int getNbOutputs() const override
	{
		std::cout << "priorbox getNbOutputs" << std::endl;
		return 1;
	}
	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
	{
		std::cout << "PriorBox getOutputDimensions" << std::endl;
		/* std::cout << "index: " << index << std::endl;
		std::cout << "nbInputDims: " << nbInputDims << std::endl;
		assert(nbInputDims == 2);
		assert(index <= 1);
		assert(inputs[index].nbDims == 3);
		for (int i = 1; i < inputs[index].nbDims; i++)
		{
			if(i == CatAxis) continue;
			assert(inputs[0].d[i] == inputs[index].d[i]);
		}
		
		DimsCHW dims(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
		for (int i = 1; i < nbInputDims; i++)
		{
			dims.d[CatAxis] += inputs[i].d[CatAxis];
		} */
		assert(nbInputDims == 2);

		std::cout << inputs[0].d[1] << std::endl;
		std::cout << inputs[0].d[2] << std::endl;
		std::cout << num_priors_ << std::endl;
		
		return DimsCHW(2, inputs[0].d[1] * inputs[0].d[2] * num_priors_ * 4, 1);
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
		/******param*****/
		float variance_[4+1] = {0.10000000149f, 0.10000000149f, 0.20000000298f, 0.20000000298f};
	    float offset_ = 0.5f;
		bool clip_ = false;
		std::cout << "PriorBox enqueue" << std::endl;
		clock_t start_clock = clock();
		std::cout <<  clock() << std::endl;
		
	  //const int layer_width_ = bottom[0]->width();
	  //const int layer_height_ = bottom[0]->height();
	  //int img_width_, img_height_;
	  /*if (img_h_ == 0 || img_w_ == 0) {
		img_width_ = bottom[1]->width();
		img_height_ = bottom[1]->height();
	  } else {
		img_width_ = img_w_;
		img_height_ = img_h_;
	  }*/
	  float step_w, step_h;
	  /*if (step_w_ == 0 || step_h_ == 0) {
		step_w = static_cast<float>(img_width_) / layer_width_;
		step_h = static_cast<float>(img_height_) / layer_height_;
	  } else {
		step_w = step_w_;
		step_h = step_h_;
	  }*/
	  step_w = step_h = step_;
	  //float* top_data = top[0]->mutable_cpu_data();
	  int dim = layer_height_ * layer_width_ * num_priors_ * 4;
      //std::cout << "dim " << dim << std::endl;
	  float* top_data = new float[2*dim+1];//channel=2
	  int idx = 0;
	  //if(std::isinf(top_data[dim-1] )) std::cout << "before " << std::endl;
	  for (int h = 0; h < layer_height_; ++h) {
		for (int w = 0; w < layer_width_; ++w) {
		  float center_x = (w + offset_) * step_w;
		  float center_y = (h + offset_) * step_h;
		  float box_width, box_height;
		  for (int s = 0; s < min_sizes_.size(); ++s) {
			int min_size_ = min_sizes_[s];
			// first prior: aspect_ratio = 1, size = min_size
			box_width = box_height = min_size_;
			// xmin
			top_data[idx++] = (center_x - box_width / 2.) / img_width_;
			// ymin
			top_data[idx++] = (center_y - box_height / 2.) / img_height_;
			//if(std::isinf(top_data[idx] )) std::cout << idx << "inf " << box_height << img_height_ << std::endl;
			// xmax
			top_data[idx++] = (center_x + box_width / 2.) / img_width_;
			// ymax
			top_data[idx++] = (center_y + box_height / 2.) / img_height_;
			//if(std::isinf(top_data[idx-1] )) std::cout << idx << "inf " << box_height << img_height_ << std::endl;

			if (max_sizes_.size() > 0) {
			  CHECK_EQ(min_sizes_.size(), max_sizes_.size());
			  int max_size_ = max_sizes_[s];
			  // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
			  box_width = box_height = sqrt(min_size_ * max_size_);
			  // xmin
			  top_data[idx++] = (center_x - box_width / 2.) / img_width_;
			  // ymin
			  top_data[idx++] = (center_y - box_height / 2.) / img_height_;
			  //if(std::isinf(top_data[idx] )) std::cout << idx << "inf " << box_height << img_height_ << std::endl;
			  // xmax
			  top_data[idx++] = (center_x + box_width / 2.) / img_width_;
			  // ymax
			  top_data[idx++] = (center_y + box_height / 2.) / img_height_;
			  //if(std::isinf(top_data[idx-1] )) std::cout << idx << "inf " << box_height << img_height_ << std::endl;
			}

			// rest of priors
			for (int r = 0; r < aspect_ratios_.size(); ++r) {
			  float ar = aspect_ratios_[r];
			  if (fabs(ar - 1.) < 1e-6) {
				continue;
			  }
			  box_width = min_size_ * sqrt(ar);
			  box_height = min_size_ / sqrt(ar);
			  //if(sqrt(ar)<0.1f) std::cout << ar << sqrt(ar) << std::endl;
			  // xmin
			  top_data[idx++] = (center_x - box_width / 2.) / img_width_;
			  // ymin
			  top_data[idx++] = (center_y - box_height / 2.) / img_height_;
			  //if(std::isinf(top_data[idx] )) std::cout << idx << "inf " << box_height << img_height_ << std::endl;
			  // xmax
			  top_data[idx++] = (center_x + box_width / 2.) / img_width_;
			  // ymax
			  top_data[idx++] = (center_y + box_height / 2.) / img_height_;
			  //if(std::isinf(top_data[idx-1] )) std::cout << idx << "inf " << box_height << img_height_ << std::endl;
			}
			//std::cout << "idx " << idx << std::endl;
		  }
		}
	  }
	  //if(std::isinf(top_data[dim-1] )) std::cout << "after " << std::endl;

	  //std::cout << "idx " << idx << std::endl;
	  // clip the prior's coordidate such that it is within [0, 1]
	  if (clip_) {
		for (int d = 0; d < dim; ++d) {
		  top_data[d] = std::min<float>(std::max<float>(top_data[d], 0.), 1.);
		}
	  }
	  // set the variance.
	  //top_data += top[0]->offset(0, 1);
	  float *top_dataC2 = top_data + dim;
	  /*if (variance_.size() == 1) {
		caffe_set<Dtype>(dim, Dtype(variance_[0]), top_data);
	  } else {*/
		int count = 0;
		for (int h = 0; h < layer_height_; ++h) {
		  for (int w = 0; w < layer_width_; ++w) {
			for (int i = 0; i < num_priors_; ++i) {
			  for (int j = 0; j < 4; ++j) {
				top_dataC2[count] = variance_[j];
				++count;
			  }
			}
		  }
		}
	  //std::cout << "count " << count << std::endl;
	  //}
	  newCHECK(cudaMemcpyAsync(outputs[0], top_data, 2 * dim * sizeof(float), cudaMemcpyHostToDevice, stream));
	  /*std::cout << "bbox " << std::endl;
	  for(int i=0; i < 100; i++) std::cout << top_data[i] << " ";std::cout << top_data[dim-1] << std::endl;
	  std::cout << "variance " << std::endl;
	  for(int i=dim; i < dim+100; i++) std::cout << top_data[i] << " ";std::cout << top_data[2*dim-1] << std::endl;*/
  	  clock_t clock_sum = 0;
	  clock_sum += (clock() - start_clock);
	  std::cout <<  clock() << std::endl;
	  std::cout <<  static_cast<double>( clock_sum ) / CLOCKS_PER_SEC << std::endl;
	  delete[] top_data;
	  return 0;
	}

	size_t getSerializationSize() override
	{
		return sizeof(int)*5;
	}

	void serialize(void* buffer) override
	{
		char* d = reinterpret_cast<char*>(buffer), *a = d;

		write(d, ID);
		write(d, layer_height_);
		write(d, layer_width_);
		write(d, img_height_);
		write(d, img_width_);		

		assert(d == a + getSerializationSize());
	}

	void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int)	override
	{
		assert(nbInputs == 2);
		assert(nbOutputs == 1);
		assert(inputs[0].nbDims == 3);
		assert(inputs[1].nbDims == 3);
		
		layer_height_ = inputs[0].d[1];
		layer_width_ = inputs[0].d[2];
		
		img_height_ = inputs[1].d[1];
		img_width_ = inputs[1].d[2];
	}

private:
	int ID;
	//int layer_width_, layer_height_, img_width_, img_height_, num_priors_, offset_, min_sizes_, max_sizes_, aspect_ratios_, *variance_ = NULL;
	int layer_width_, layer_height_, img_width_, img_height_, num_priors_, step_;
	std::vector<int> min_sizes_, max_sizes_;
	std::vector<float> aspect_ratios_;

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
	
protected:	

};
