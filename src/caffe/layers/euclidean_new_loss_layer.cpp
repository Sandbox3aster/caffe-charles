#include <vector>
#include <iostream>

#include "caffe/layers/euclidean_new_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {

template <typename Dtype>
void EuclideanNewLossLayer<Dtype>::LayerSetUp(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LOG(INFO) <<"setup  1     ======="<< std::endl;
	// LossLayers have a non-zero (1) loss by default.
   vector<int> shape(0);
   top[0]->Reshape(shape);
   LOG(INFO)<<"setup  2     ======="<< std::endl;
	if (this->layer_param_.loss_weight_size() == 0) {
		this->layer_param_.add_loss_weight(Dtype(1));
	}
}

template <typename Dtype>
void EuclideanNewLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //LOG(INFO)<<"reshape  2     ======="<< std::endl;
  vector<int> shape(1);
  shape[0] = 1;
  top[0]->Reshape(shape);
  //LOG(INFO)<<"reshape  2     ======="<< std::endl;
  int net_size = this->layer_param_.euclidean_new_loss_param().net_size();
  int label_index = net_size-1;
  //LOG(INFO)<<" net_size "<< net_size<<" label_index "<<label_index<< std::endl;
  for (int i = 0; i < net_size-3; ++i)
  {
//	  LOG(INFO)<<"i = "<<i<<std::endl;
	  CHECK_EQ(bottom[i]->width(), bottom[i+1]->width());
//	  LOG(INFO)<<" width   "<< std::endl;
	  CHECK_EQ(bottom[i]->height(), bottom[i+1]->height());
	  CHECK_EQ(bottom[i]->channels(), bottom[i + 1]->channels());
	  CHECK_EQ(bottom[i]->num(), bottom[i + 1]->num());
  }
  CHECK_EQ(bottom[label_index]->width(), 1);
  CHECK_EQ(bottom[label_index]->height(), 1);

  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
//  LOG(INFO)<<" diff_ ======================" << std::endl;
  diff_.ReshapeLike(*bottom[0]);
  //LOG(INFO) << "diff_     ===========" << std::endl;
}

template <typename Dtype>
void EuclideanNewLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    NOT_IMPLEMENTED;
}

template <typename Dtype>
void EuclideanNewLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	
	NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanNewLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanNewLossLayer);
REGISTER_LAYER_CLASS(EuclideanNewLoss);

}  // namespace caffe
