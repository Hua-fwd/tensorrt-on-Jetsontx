#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

#include "plugin/detectionOutputPlugin.hpp"


template <int num_class>
void DetectionOutput<num_class>::Forward_gpu(int batchSize, const void*const *inputs, void** outputs, void* workspace, cudaStream_t stream) {
  const float* loc_data = reinterpret_cast<const float*>(inputs[0]);
  const float* conf_data = reinterpret_cast<const float*>(inputs[1]);
  const float* prior_data = reinterpret_cast<const float*>(inputs[2]);
  const int num = batchSize;
  
  bool share_location_ = true;

  // Decode predictions.
  //float* bbox_data = bbox_preds_.mutable_gpu_data();
  float* bbox_data;
  newCHECK(cudaMalloc(&bbox_data, d1 * sizeof(float)));
  //const int loc_count = bbox_preds_.count();
  const int loc_count = d1;
  const bool clip_bbox = false;
  int code_type_ = 2;
  DecodeBBoxesGPU<float>(loc_count, loc_data, prior_data, code_type_,
      variance_encoded_in_target_, num_priors_, share_location_,
      num_loc_classes_, background_label_id_, clip_bbox, bbox_data);
  // Retrieve all decoded location predictions.
  float* bbox_cpu_data = new float[d1];
  if (!share_location_) {
    //float* bbox_permute_data = bbox_permute_.mutable_gpu_data();
	float* bbox_permute_data;
	newCHECK(cudaMalloc(&bbox_permute_data, d1 * sizeof(float)));
    PermuteDataGPU<float>(loc_count, bbox_data, num_loc_classes_, num_priors_,
        4, bbox_permute_data);
    //bbox_cpu_data = bbox_permute_.cpu_data();
	//bbox_cpu_data = bbox_permute_data;
	newCHECK(cudaMemcpyAsync(bbox_cpu_data, bbox_permute_data, d1 * sizeof(float), cudaMemcpyDeviceToHost, stream));
	newCHECK(cudaFree(bbox_permute_data));
  } else {
    //bbox_cpu_data = bbox_preds_.cpu_data();
	//bbox_cpu_data = bbox_data;
	newCHECK(cudaMemcpyAsync(bbox_cpu_data, bbox_data, d1 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  }

  // Retrieve all confidences.
  //float* conf_permute_data = conf_permute_.mutable_gpu_data();
  float* conf_permute_data;
  newCHECK(cudaMalloc(&conf_permute_data, d2 * sizeof(float)));
  PermuteDataGPU<float>(d2, conf_data,
      num_classes_, num_priors_, 1, conf_permute_data);
  //const float* conf_cpu_data = conf_permute_.cpu_data();
  //const float* conf_cpu_data = conf_permute_data;
  float* conf_cpu_data = new float[d2];
  newCHECK(cudaMemcpyAsync(conf_cpu_data, conf_permute_data, d2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
  int num_kept = 0;
  vector<map<int, vector<int> > > all_indices;
  for (int i = 0; i < num; ++i) {
    map<int, vector<int> > indices;
    int num_det = 0;
    const int conf_idx = i * num_classes_ * num_priors_;
    int bbox_idx;
    if (share_location_) {
      bbox_idx = i * num_priors_ * 4;
    } else {
      bbox_idx = conf_idx * 4;
    }
    for (int c = 0; c < num_classes_; ++c) {
      if (c == background_label_id_) {
        // Ignore background class.
        continue;
      }
      const float* cur_conf_data = conf_cpu_data + conf_idx + c * num_priors_;
      const float* cur_bbox_data = bbox_cpu_data + bbox_idx;
      if (!share_location_) {
        cur_bbox_data += c * num_priors_ * 4;
      }

      ApplyNMSFast(cur_bbox_data, cur_conf_data, num_priors_,
          confidence_threshold_, nms_threshold_, eta_, top_k_, &(indices[c]));
      num_det += indices[c].size();
    }
    if (keep_top_k_ > -1 && num_det > keep_top_k_) {
      vector<pair<float, pair<int, int> > > score_index_pairs;
      for (map<int, vector<int> >::iterator it = indices.begin();
           it != indices.end(); ++it) {
        int label = it->first;
        const vector<int>& label_indices = it->second;
        for (int j = 0; j < label_indices.size(); ++j) {
          int idx = label_indices[j];
          float score = conf_cpu_data[conf_idx + label * num_priors_ + idx];
          score_index_pairs.push_back(std::make_pair(
                  score, std::make_pair(label, idx)));
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


  /*vector<int> top_shape(2, 1);
  top_shape.push_back(num_kept);
  top_shape.push_back(7);*/
  //float* top_data = reinterpret_cast<float*>(outputs[0]);
  float* top = new float[keep_top_k_*7];
  float* top_data = top;
  if (num_kept == 0) {
    LOG(INFO) << "Couldn't find any detections";
    /*top_shape[2] = num;
    top[0]->Reshape(top_shape);
    top_data = top[0]->mutable_cpu_data();*/
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
    const int conf_idx = i * num_classes_ * num_priors_;
    int bbox_idx;
    if (share_location_) {
      bbox_idx = i * num_priors_ * 4;
    } else {
      bbox_idx = conf_idx * 4;
    }
    for (map<int, vector<int> >::iterator it = all_indices[i].begin();
         it != all_indices[i].end(); ++it) {
      int label = it->first;
      vector<int>& indices = it->second;
      const float* cur_conf_data =
        conf_cpu_data + conf_idx + label * num_priors_;
      const float* cur_bbox_data = bbox_cpu_data + bbox_idx;
      if (!share_location_) {
        cur_bbox_data += label * num_priors_ * 4;
      }
      for (int j = 0; j < indices.size(); ++j) {
        int idx = indices[j];
        top_data[count * 7] = i;
        top_data[count * 7 + 1] = label;
        top_data[count * 7 + 2] = cur_conf_data[idx];
        for (int k = 0; k < 4; ++k) {
          top_data[count * 7 + 3 + k] = cur_bbox_data[idx * 4 + k];
        }
        ++count;
      }
    }
  }
  if(count < keep_top_k_) 
  {
	top_data[count * 7] = -1;
	count ++;
  }
  newCHECK(cudaMemcpyAsync(outputs[0], top, (count * 7) * sizeof(float), cudaMemcpyHostToDevice, stream));

  delete[] bbox_cpu_data;
  delete[] conf_cpu_data;
  delete[] top;
  newCHECK(cudaFree(bbox_data));
  newCHECK(cudaFree(conf_permute_data));

}

