#include <vector>

#include "caffe/layers/euclidean_new_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanNewLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  int net_size = this->layer_param_.euclidean_new_loss_param().net_size();
  int label_index = net_size - 2;
  int random_index = net_size - 1;
  int bottom_used_idx = static_cast<int>(bottom[random_index]->cpu_data()[0]);  
  caffe_gpu_sub(
	  count,
      bottom[bottom_used_idx]->gpu_data(),
      bottom[label_index]->gpu_data(),
      diff_.mutable_gpu_data());

  Dtype dot;
/*
  LOG(INFO)<<"bottom[bottom_used_idx]->gpu_data()"<< "  "<<std::endl;
  for (int idx_b=0;idx_b<bottom[bottom_used_idx]->num();idx_b++)
	LOG(INFO)<<" "<<bottom[bottom_used_idx]->cpu_data()[idx_b]<<" ";
  LOG(INFO)<<"bottom[label_index]->gpu_data() "<< "   "<<std::endl;
  for (int idx_l=0;idx_l<bottom[label_index]->num();idx_l++)
	LOG(INFO)<<" "<<bottom[label_index]->cpu_data()[idx_l]<<" ";
  LOG(INFO)<<"diff_.gpu_data()"<< "   "<<std::endl;
  for (int idx_t=0;idx_t<diff_.num();idx_t++)
	LOG(INFO)<<" "<<diff_.cpu_data()[idx_t]<<" ";
*/
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / bottom[bottom_used_idx]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanNewLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	
	int net_size = this->layer_param_.euclidean_new_loss_param().net_size();
	int bottom_size = net_size - 2;
	int random_index = net_size-1;
    int bottom_used_idx = static_cast<int>(bottom[random_index]->cpu_data()[0]);
	  for (int i = 0; i < bottom_size; ++i) {
		  if (i != bottom_used_idx)
		  {
			  Dtype alpha = 0.0;
			  caffe_gpu_set(
				  bottom[i]->count(),              // count
				  alpha,							// alpha
				  bottom[i]->mutable_gpu_diff());  // Y
		  }
		  else 
		  {
              const Dtype sign = 1;
			  const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
			  caffe_gpu_axpby(
				  bottom[i]->count(),              // count
				  alpha,                              // alpha
				  diff_.gpu_data(),                   // a
				  Dtype(0),                           // beta
				  bottom[i]->mutable_gpu_diff());  // bs
		  }
	  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanNewLossLayer);

}  // namespace caffe
