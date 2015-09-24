// Copyright 2014 Tobias Domhan

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include <algorithm>
#include <cmath>

using std::max;
int sigm_iter_num = 1;

namespace caffe {

template<typename Dtype>
void SigmoidLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	NeuronLayer<Dtype>::SetUp(bottom, top);
	sigm_decay = this->layer_param_.sigm_decay();
	sigm_para = this->layer_param_.sigm_para();
	iter_decay = this->layer_param_.iter_decay();
}

template<typename Dtype>
inline Dtype sigmoid(Dtype x, float sigm) {
	//return 1. / (1. + exp(-5 * x));
	return 2. / (1. + exp(-sigm * x)) - 1.;
}

template<typename Dtype>
void SigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = (*top)[0]->mutable_cpu_data();
	const int count = bottom[0]->count();
	if (sigm_iter_num % iter_decay == 0) {
		sigm_para = sigm_para * sigm_decay;
	}
	for (int i = 0; i < count; ++i) {
		top_data[i] = sigmoid(bottom_data[i], sigm_para);
	}
}

template<typename Dtype>
Dtype SigmoidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
	if (propagate_down) {
		const Dtype* bottom_data = (*bottom)[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
		const int count = (*bottom)[0]->count();
		for (int i = 0; i < count; ++i) {
			Dtype sigmoid_x = sigmoid(bottom_data[i], sigm_para);
			bottom_diff[i] = 0.5 * sigm_para * top_diff[i] * (1. + sigmoid_x)
					* (1. - sigmoid_x);
		}
	}
	sigm_iter_num++;
	return Dtype(0);
}

template<typename Dtype>
__device__  inline Dtype sigmoid_gpu(Dtype x, float sigm) {
	//return 1. / (1. + exp(-5 * x));
	return 2. / (1. + exp(-sigm * x)) - 1.;
}

template<typename Dtype>
__global__ void SigmoidForward(const int n, const Dtype* in, Dtype* out,
		float sigm) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n) {
		out[index] = sigmoid_gpu(in[index], sigm);
	}
}

template<typename Dtype>
void SigmoidLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = (*top)[0]->mutable_gpu_data();
	const int count = bottom[0]->count();

	if (sigm_iter_num % iter_decay == 0) {
		sigm_para = sigm_para * sigm_decay;
	}

	LOG(INFO) << "sigm_para :" << sigm_para;
	SigmoidForward<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			count, bottom_data, top_data, sigm_para);
	CUDA_POST_KERNEL_CHECK;
	// << " count: " << count << " bottom_data: "
	//     << (unsigned long)bottom_data << " top_data: " << (unsigned long)top_data
	//     << " blocks: " << CAFFE_GET_BLOCKS(count)
	//     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template<typename Dtype>
__global__ void SigmoidBackward(const int n, const Dtype* in_diff,
		const Dtype* in_data, Dtype* out_diff, float sigm) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n) {
		Dtype sigmoid_x = sigmoid_gpu(in_data[index], sigm);
		//out_diff[index] = 5 * in_diff[index] * sigmoid_x * (1 - sigmoid_x);
		out_diff[index] = 0.5 * sigm * in_diff[index] * (1 + sigmoid_x)
				* (1 - sigmoid_x);
	}
}

template<typename Dtype>
Dtype SigmoidLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
	if (propagate_down) {
		const Dtype* bottom_data = (*bottom)[0]->gpu_data();
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
		const int count = (*bottom)[0]->count();
		SigmoidBackward<Dtype> <<<CAFFE_GET_BLOCKS(count),
				CAFFE_CUDA_NUM_THREADS>>>(count, top_diff, bottom_data,
				bottom_diff, sigm_para);
		CUDA_POST_KERNEL_CHECK;
	}
	sigm_iter_num++;
	return Dtype(0);
}

INSTANTIATE_CLASS(SigmoidLayer);

}  // namespace caffe
