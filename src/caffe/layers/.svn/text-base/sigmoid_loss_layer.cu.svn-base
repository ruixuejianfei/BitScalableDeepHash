// Copyright 2013 Yangqing Jia

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template<typename Dtype>
void SigmoidWithLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	CHECK_EQ(bottom.size(), 2)<< "SigmoidLoss Layer takes a single blob as input.";
	CHECK_EQ(top->size(), 0) << "SigmoidLoss Layer takes no blob as output.";
	sigmoid_bottom_vec_.clear();
	sigmoid_bottom_vec_.push_back(bottom[0]);
	sigmoid_top_vec_.push_back(&prob_);
	sigmoid_layer_->SetUp(sigmoid_bottom_vec_, &sigmoid_top_vec_);
};

template<typename Dtype>
void SigmoidWithLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
	// The forward pass computes the sigmoid prob values.
	sigmoid_bottom_vec_[0] = bottom[0];
	sigmoid_layer_->Forward(sigmoid_bottom_vec_, &sigmoid_top_vec_);
}

template<typename Dtype>
void SigmoidWithLossLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
	// The forward pass computes the sigmoid prob values.
	sigmoid_bottom_vec_[0] = bottom[0];
	sigmoid_layer_->Forward(sigmoid_bottom_vec_, &sigmoid_top_vec_);
}

template<typename Dtype>
Dtype SigmoidWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
	// First, compute the diff
	Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
	const Dtype* prob_data = prob_.cpu_data();
	memcpy(bottom_diff, prob_data, sizeof(Dtype) * prob_.count());
	const Dtype* label = (*bottom)[1]->cpu_data();
	int num = prob_.num();
	int dim = prob_.count() / num;
	Dtype loss = 0;
	// zhu
	// calculate loss for each input sample
	this->losses_.clear();
	for (int i = 0; i < num; ++i) {
		this->losses_.push_back(Dtype(0));
		for (int j = 0; j < dim; ++j) {
			if (label[i * dim + j] == 1)
				bottom_diff[i * dim + j] -= 1;
			if (label[i * dim + j] == 1) {
//				loss += -log(max(prob_data[i * dim + j], FLT_MIN));
				this->losses_[i] += -log(max(prob_data[i * dim + j], FLT_MIN));
			} else {
//				loss += -log(max(1 - prob_data[i * dim + j], FLT_MIN));
				this->losses_[i] += -log(
						max(1 - prob_data[i * dim + j], FLT_MIN));
			}
		}
		loss += this->losses_[i];
	}
	// Scale down gradient
	caffe_scal(prob_.count(), Dtype(1) / num, bottom_diff);
	return loss / num;
}

template<typename Dtype>
Dtype SigmoidWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
	// TODO(Yangqing): implement the GPU version of sigmoid.
	return Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_CLASS(SigmoidWithLossLayer);

}  // namespace caffe
