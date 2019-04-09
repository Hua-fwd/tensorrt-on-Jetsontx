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
#include <map>
#include <ctime>

#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "util/math_functions.hpp"
#include "util/caffe.pb.h"
#include "util/bbox_util.hpp"
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
using namespace caffe;
using namespace std;

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

template <int num_class>
class DetectionOutput : public IPlugin
{
public:
	DetectionOutput() {
		keep_top_k_ = 200;
	}
	DetectionOutput(const void* buffer, size_t size)
	{
		const char* d = reinterpret_cast<const char*>(buffer), *a = d;
		d1 = read<int>(d);
		d2 = read<int>(d);
		d3 = read<int>(d);
		num_priors_ = read<int>(d);
		
		initParam();
		assert(d == a + size);
	}

	int getNbOutputs() const override
	{
		return 1;
	}
	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
	{
		//std::cout << "index: " << index << std::endl;
		//std::cout << "nbInputDims: " << nbInputDims << std::endl;
		/*assert(nbInputDims == 2);
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
		}
		return dims;*/
		return DimsCHW( 1, keep_top_k_, 7);
	}
	
	void initParam()
	{
		  num_classes_ = num_class;
		  share_location_ = true;
		  num_loc_classes_ = share_location_ ? 1 : num_classes_;
		  background_label_id_ = 0;
		  //code_type_ = detection_output_param.code_type();
		  variance_encoded_in_target_ = false;
		  keep_top_k_ = 200;
		  //confidence_threshold_ = 0.00999999977648f;
		  confidence_threshold_ = 0.01f;
		  // Parameters used in nms.
		  //nms_threshold_ = 0.449999988079f;
		  nms_threshold_ = 0.45f;
		  CHECK_GE(nms_threshold_, 0.) << "nms_threshold must be non negative.";
		  eta_ = 1.0f;
		  CHECK_GT(eta_, 0.);
		  CHECK_LE(eta_, 1.);
		  top_k_ = 400;
	
		  need_save_ = false;
		  
		  name_count_ = 0;
		  //bbox_preds_.ReshapeLike(*(bottom[0]));
		  /*if (!share_location_) {
			bbox_permute_.ReshapeLike(*(bottom[0]));
		  }*/
		  //conf_permute_.ReshapeLike(*(bottom[1]));
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
	
	void Forward_gpu(int batchSize, const void*const *inputs, void** outputs, void* workspace, cudaStream_t stream);

	void Forward_cpu(int batchSize, const void*const *inputs, void** outputs, void* workspace, cudaStream_t stream)
	{
		//bool need_save_ = false;
		  bool share_location_ = true;
		  //const Dtype* loc_data = bottom[0]->cpu_data();
		  //const Dtype* conf_data = bottom[1]->cpu_data();
		  //const Dtype* prior_data = bottom[2]->cpu_data();
		  float *loc_data = new float[d1];
		  float *conf_data = new float[d2];
		  float *prior_data = new float[d3];
		  newCHECK(cudaMemcpyAsync(loc_data, inputs[0], d1 * sizeof(float), cudaMemcpyDeviceToHost, stream));
		  newCHECK(cudaMemcpyAsync(conf_data, inputs[1], d2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
		  newCHECK(cudaMemcpyAsync(prior_data, inputs[2], d3 * sizeof(float), cudaMemcpyDeviceToHost, stream));
		/*std::cout << "d1 " << d1 << std::endl;
		std::cout << "d2 " << d2 << std::endl;
		std::cout << "d3 " << d3 << std::endl;

		  for(int i=0; i < 100; i++) std::cout << loc_data[i] << " ";std::cout << loc_data[d1-1] << std::endl;
		  std::cout << std::endl;
		  for(int i=0; i < 100; i++) std::cout << conf_data[i] << " ";std::cout << conf_data[d2-1] << std::endl;
		  std::cout << std::endl;
		  for(int i=0; i < 100; i++) std::cout << prior_data[i] << " ";std::cout << prior_data[d3-1] << std::endl;
		  std::cout << std::endl;*/

		vector<float> tem(conf_data, conf_data+d2-1);
		//tem.assign(scores.begin(), scores.end());  
		sort(tem.begin(), tem.end(), greater<float>());
		//for(int i = 0; i < 100; i++) std::cout << tem[i] << std::endl;

		  /*for(int i=0; i < 100; i++) std::cout << conf_data[i] << " ";
		  std::cout << std::endl;*/
		  const int num = batchSize;
		  assert(num == 1);

		  // Retrieve all location predictions.
		  vector<LabelBBox> all_loc_preds;
		  GetLocPredictions(loc_data, num, num_priors_, num_loc_classes_,
							share_location_, &all_loc_preds);

		  // Retrieve all confidences.
		  vector<map<int, vector<float> > > all_conf_scores;
		  GetConfidenceScores(conf_data, num, num_priors_, num_classes_,
							  &all_conf_scores);

		  // Retrieve all prior bboxes. It is same within a batch since we assume all
		  // images in a batch are of same dimension.
		  vector<NormalizedBBox> prior_bboxes;
		  vector<vector<float> > prior_variances;
		  GetPriorBBoxes(prior_data, num_priors_, &prior_bboxes, &prior_variances);

		  // Decode all loc predictions to bboxes.
		  vector<LabelBBox> all_decode_bboxes;
		  const bool clip_bbox = false;
		  int code_type_ = 2;
		  /*std::cout << prior_bboxes[0].xmax() << std::endl;
		  std::cout << prior_bboxes[0].xmin() << std::endl;*/
		  //float prior_width = prior_bbox.xmax() - prior_bbox.xmin();
		  DecodeBBoxesAll(all_loc_preds, prior_bboxes, prior_variances, num,
						  share_location_, num_loc_classes_, background_label_id_,
						  code_type_, variance_encoded_in_target_, clip_bbox,
						  &all_decode_bboxes);

		  int num_kept = 0;
		  vector<map<int, vector<int> > > all_indices;
		  for (int i = 0; i < num; ++i) {
			const LabelBBox& decode_bboxes = all_decode_bboxes[i];
			const map<int, vector<float> >& conf_scores = all_conf_scores[i];

			map<int, vector<int> > indices;
			int num_det = 0;
			for (int c = 0; c < num_classes_; ++c) {
			  if (c == background_label_id_) {
				// Ignore background class.
				continue;
			  }
			  if (conf_scores.find(c) == conf_scores.end()) {
				// Something bad happened if there are no predictions for current label.
				LOG(FATAL) << "Could not find confidence predictions for label " << c;
			  }
			  const vector<float>& scores = conf_scores.find(c)->second;
			  int label = share_location_ ? -1 : c;
			  if (decode_bboxes.find(label) == decode_bboxes.end()) {
				// Something bad happened if there are no predictions for current label.
				LOG(FATAL) << "Could not find location predictions for label " << label;
				continue;
			  }
			  const vector<NormalizedBBox>& bboxes = decode_bboxes.find(label)->second;
			  ApplyNMSFast(bboxes, scores, confidence_threshold_, nms_threshold_, eta_,
				  top_k_, &(indices[c]));
			  num_det += indices[c].size();
			}
			if (keep_top_k_ > -1 && num_det > keep_top_k_) {
			  vector<pair<float, pair<int, int> > > score_index_pairs;
			  for (map<int, vector<int> >::iterator it = indices.begin();
				   it != indices.end(); ++it) {
				int label = it->first;
				const vector<int>& label_indices = it->second;
				if (conf_scores.find(label) == conf_scores.end()) {
				  // Something bad happened for current label.
				  LOG(FATAL) << "Could not find location predictions for " << label;
				  continue;
				}
				const vector<float>& scores = conf_scores.find(label)->second;
				
				for (int j = 0; j < label_indices.size(); ++j) {
				  int idx = label_indices[j];
				  CHECK_LT(idx, scores.size());
				  score_index_pairs.push_back(std::make_pair(
						  scores[idx], std::make_pair(label, idx)));
				}
			  }
			  // Keep top k results per image.
			  std::sort(score_index_pairs.begin(), score_index_pairs.end(),
						SortScorePairDescend<pair<int, int> >);
			  score_index_pairs.resize(keep_top_k_);
			  // Store the new indices.
			  map<int, vector<int> > new_indices;
			  for (int j = 0; j < score_index_pairs.size(); ++j) {
				int label = score_index_pairs[j].second.first;
				int idx = score_index_pairs[j].second.second;
				new_indices[label].push_back(idx);
			  }
			  all_indices.push_back(new_indices);
			  num_kept += keep_top_k_;
			} else {
			  all_indices.push_back(indices);
			  num_kept += num_det;
			}
		  }

		  vector<int> top_shape(2, 1);
		  top_shape.push_back(num_kept);
		  top_shape.push_back(7);

		  float* top = new float[keep_top_k_*7];
		  float* top_data = top;
		  if (num_kept == 0) {
			LOG(INFO) << "Couldn't find any detections";
			top_shape[2] = num;
			//top[0]->Reshape(top_shape);
			//top_data = top[0]->mutable_cpu_data();
			
			caffe_set<float>(keep_top_k_*7, -1, top_data);
			// Generate fake results per image.
			for (int i = 0; i < num; ++i) {
			  top_data[0] = i;
			  top_data += 7;
			}
		  } else {
			//top[0]->Reshape(top_shape);
			//top_data = top[0]->mutable_cpu_data();
		  }

		  int count = 0;
		  //boost::filesystem::path output_directory(output_directory_);
		  for (int i = 0; i < num; ++i) {
			const map<int, vector<float> >& conf_scores = all_conf_scores[i];
			const LabelBBox& decode_bboxes = all_decode_bboxes[i];
			for (map<int, vector<int> >::iterator it = all_indices[i].begin();
				 it != all_indices[i].end(); ++it) {
			  int label = it->first;
			  //std::cout << "label " << label << std::endl;
			  if (conf_scores.find(label) == conf_scores.end()) {
				// Something bad happened if there are no predictions for current label.
				LOG(FATAL) << "Could not find confidence predictions for " << label;
				continue;
			  }
			  const vector<float>& scores = conf_scores.find(label)->second;
			  int loc_label = share_location_ ? -1 : label;
			  if (decode_bboxes.find(loc_label) == decode_bboxes.end()) {
				// Something bad happened if there are no predictions for current label.
				LOG(FATAL) << "Could not find location predictions for " << loc_label;
				continue;
			  }
			  const vector<NormalizedBBox>& bboxes =
				  decode_bboxes.find(loc_label)->second;
			  vector<int>& indices = it->second;
			  
			  for (int j = 0; j < indices.size(); ++j) {
				int idx = indices[j];
				top_data[count * 7] = i;
				top_data[count * 7 + 1] = label;
				top_data[count * 7 + 2] = scores[idx];
				//std::cout << "scores[idx] " << scores[idx] << std::endl;
				const NormalizedBBox& bbox = bboxes[idx];
				top_data[count * 7 + 3] = bbox.xmin();
				top_data[count * 7 + 4] = bbox.ymin();
				top_data[count * 7 + 5] = bbox.xmax();
				top_data[count * 7 + 6] = bbox.ymax();
				
				++count;
			  }
			//std::cout << "count " << count <<std::endl;
			}
			
		  }
		  /*if (visualize_) {
		#ifdef USE_OPENCV
			vector<cv::Mat> cv_imgs;
			this->data_transformer_->TransformInv(bottom[3], &cv_imgs);
			vector<cv::Scalar> colors = GetColors(label_to_display_name_.size());
			VisualizeBBox(cv_imgs, top[0], visualize_threshold_, colors,
				label_to_display_name_, save_file_);
		#endif  // USE_OPENCV
		  }*/
		std::cout << "detec num " << count << std::endl;
		/*for(int i = 0; i < keep_top_k_*7; i++)
		{
			std::cout << top[i] << ' ';
			if((i+1)%7==0) std::cout << std::endl;
		}*/
		//std::cout << std::endl;
		/*if(count == 0)
		{
			count++;
			top[count*7] = -1;
			count ++;
		}
		else */if(count < keep_top_k_) 
		{
			top[count*7] = -1;
			//std::cout << top[count*7] << std::endl;
			count ++;
		}
		else if(count > keep_top_k_)
		{
			count = keep_top_k_;
		}
		newCHECK(cudaMemcpyAsync(outputs[0], top, (count * 7) * sizeof(float), cudaMemcpyHostToDevice, stream));
		delete []loc_data;
		delete []conf_data;
		delete []prior_data;
		delete []top;
	}

	// currently it is not possible for a plugin to execute "in place". Therefore we memcpy the data from the input to the output buffer
	int enqueue(int batchSize, const void*const *inputs, void** outputs, void* workspace, cudaStream_t stream) override
	{
		std::cout << "detetc enqueue" << std::endl;
		clock_t start_clock_loc = clock();
		std::cout <<  clock() << std::endl;
		Forward_gpu(batchSize, inputs, outputs, workspace, stream);
  	  clock_t clock_sum = 0;
	  clock_sum += (clock() - start_clock_loc);
	  std::cout <<  clock() << std::endl;
	  std::cout <<  static_cast<double>( clock_sum ) / CLOCKS_PER_SEC << std::endl;


		return 0;
	}

	size_t getSerializationSize() override
	{
		return sizeof(int)*4;
	}

	void serialize(void* buffer) override
	{
		char* d = reinterpret_cast<char*>(buffer), *a = d;

		write(d, d1);
		write(d, d2);
		write(d, d3);
		write(d, num_priors_);

		assert(d == a + getSerializationSize());
	}

	void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int)	override
	{
		  assert(nbInputs == 3);
		  assert(inputs[0].nbDims == 3);
		  d1 = inputs[0].d[0];
		  assert(inputs[1].nbDims == 3);
		  d2 = inputs[1].d[0];
		  assert(inputs[2].nbDims == 3);
		  for(int i = 0; i < 3; i++)
			for(int j =0; j < 3; j++)
				std::cout << inputs[i].d[j] << std::endl;
		  assert(inputs[2].d[0] == 2);
		  d3 = 2*inputs[2].d[1];
		  num_priors_ = inputs[2].d[1]/4;
		  /*d1 = 3492;
		  d2 = 18337;
		  d3 = 2*3492;*/
		  
	}

private:
	int d1, d2, d3; //dim of bottoms
	float confidence_threshold_, nms_threshold_, eta_;
	int num_classes_, num_loc_classes_, background_label_id_, keep_top_k_, top_k_, name_count_;
	bool share_location_, variance_encoded_in_target_, need_save_;
	int num_priors_;
	
protected:	
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
