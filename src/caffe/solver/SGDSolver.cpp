// Copyright Yangqing Jia 2013

#include <cstdio>

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe {

// Return the current learning rate. The currently implemented learning rate
// policies are as follows:
//    - fixed: always return base_lr.
//    - step: return base_lr * gamma ^ (floor(iter / step))
//    - exp: return base_lr * gamma ^ iter
//    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
// where base_lr, gamma, step and power are defined in the solver parameter
// protocol buffer, and iter is the current iteration.
template<typename Dtype>
Dtype SGDSolver<Dtype>::GetLearningRate() {
	Dtype rate;
	const string& lr_policy = this->param_.lr_policy();
	if (lr_policy == "fixed") {
		rate = this->param_.base_lr();
	} else if (lr_policy == "step") {
		int current_step = this->iter_ / this->param_.stepsize();
		rate = this->param_.base_lr() * pow(this->param_.gamma(), current_step);
	} else if (lr_policy == "exp") {
		rate = this->param_.base_lr() * pow(this->param_.gamma(), this->iter_);
	} else if (lr_policy == "inv") {
		rate = this->param_.base_lr()
				* pow(Dtype(1) + this->param_.gamma() * this->iter_,
						-this->param_.power());
	} else {
		LOG(FATAL)<< "Unknown learning rate policy: " << lr_policy;
	}
	return rate;
}

template<typename Dtype>
void SGDSolver<Dtype>::PreSolve() {
	// Initialize the history
	vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
	history_.clear();
	for (int i = 0; i < net_params.size(); ++i) {
		const Blob<Dtype>* net_param = net_params[i].get();
		history_.push_back(
				shared_ptr<Blob<Dtype> >(
						new Blob<Dtype>(net_param->num(), net_param->channels(),
								net_param->height(), net_param->width())));
	}
}

template<typename Dtype>
void SGDSolver<Dtype>::ComputeUpdateValue() {
	vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
	vector<float>& net_params_lr = this->net_->params_lr();
	vector<float>& net_params_weight_decay = this->net_->params_weight_decay();
	// get the learning rate
	Dtype rate = GetLearningRate();
	if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
		LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
	}
	Dtype momentum = this->param_.momentum();
	Dtype weight_decay = this->param_.weight_decay();
	switch (Caffe::mode()) {
	case Caffe::CPU:
		for (int param_id = 0; param_id < net_params.size(); ++param_id) {
			// Compute the value to history, and then copy them to the blob's diff.
			Dtype local_rate = rate * net_params_lr[param_id];
			Dtype local_decay = weight_decay
					* net_params_weight_decay[param_id];
			caffe_axpby(net_params[param_id]->count(), local_rate,
					net_params[param_id]->cpu_diff(), momentum,
					history_[param_id]->mutable_cpu_data());
			if (local_decay) {
				// add weight decay
				caffe_axpy(net_params[param_id]->count(),
						local_decay * local_rate,
						net_params[param_id]->cpu_data(),
						history_[param_id]->mutable_cpu_data());
			}
			// copy
			caffe_copy(net_params[param_id]->count(),
					history_[param_id]->cpu_data(),
					net_params[param_id]->mutable_cpu_diff());
		}
		break;
	case Caffe::GPU:
		for (int param_id = 0; param_id < net_params.size(); ++param_id) {
			// Compute the value to history, and then copy them to the blob's diff.
			Dtype local_rate = rate * net_params_lr[param_id];
			Dtype local_decay = weight_decay
					* net_params_weight_decay[param_id];
			caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
					net_params[param_id]->gpu_diff(), momentum,
					history_[param_id]->mutable_gpu_data());
			if (local_decay) {
				// add weight decay
				caffe_gpu_axpy(net_params[param_id]->count(),
						local_decay * local_rate,
						net_params[param_id]->gpu_data(),
						history_[param_id]->mutable_gpu_data());
			}
			// copy
			caffe_gpu_copy(net_params[param_id]->count(),
					history_[param_id]->gpu_data(),
					net_params[param_id]->mutable_gpu_diff());
		}
		break;
	default:
		LOG(FATAL)<< "Unknown caffe mode: " << Caffe::mode();
	}
}

template<typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverState(SolverState* state) {
	state->clear_history();
	for (int i = 0; i < history_.size(); ++i) {
		// Add history
		BlobProto* history_blob = state->add_history();
		history_[i]->ToProto(history_blob);
	}
}

template<typename Dtype>
void SGDSolver<Dtype>::RestoreSolverState(const SolverState& state) {
	CHECK_EQ(state.history_size(), history_.size())<< "Incorrect length of history blobs.";
	LOG(INFO)  << "SGDSolver: restoring history";
	for (int i = 0; i < history_.size(); ++i) {
		history_[i]->FromProto(state.history(i));
	}
}

INSTANTIATE_CLASS(SGDSolver);

}  // namespace caffe
