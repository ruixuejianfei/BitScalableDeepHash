// Copyright Yangqing Jia 2013

#include <cstdio>

#include <algorithm>
#include <string>
#include <vector>
#include <sys/time.h>

#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;
using std::min;

namespace caffe {

template<typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param) :
		param_(param), net_(), test_net_() {
	// Scaffolding code
	NetParameter train_net_param;
	ReadProtoFromTextFile(param_.train_net(), &train_net_param);
	LOG(INFO)<< "Creating training net.";
	net_.reset(new Net<Dtype>(train_net_param));
	if (param_.has_test_net()) {
		LOG(INFO)<< "Creating testing net.";
		NetParameter test_net_param;
		ReadProtoFromTextFile(param_.test_net(), &test_net_param);
		test_net_.reset(new Net<Dtype>(test_net_param));
		CHECK_GT(param_.test_iter(), 0);
		CHECK_GT(param_.test_interval(), 0);
	}
	LOG(INFO)<< "Solver scaffolding done.";
}

template<typename Dtype>
void Solver<Dtype>::Solve(const char* resume_file) {
	Caffe::set_mode(Caffe::Brew(param_.solver_mode()));
	if (param_.solver_mode() && param_.has_device_id()) {
		Caffe::SetDevice(param_.device_id());
	}
	Caffe::set_phase(Caffe::TRAIN);
	LOG(INFO)<< "Solving " << net_->name();
	PreSolve();

	iter_ = 0;
	if (resume_file) {
		LOG(INFO)<< "Restoring previous solver status from " << resume_file;
		Restore(resume_file);
	}

	// For a network that is trained by the solver, no bottom or top vecs
	// should be given, and we will just provide dummy vecs.
	vector<Blob<Dtype>*> bottom_vec;
	timeval start_t, finish_t, tmp_t;
	gettimeofday(&start_t, NULL);
	gettimeofday(&tmp_t, NULL);
	int pic_counts = 0;
	int pos_triplets = 0;
	int triplets_count = 0;
	while (iter_++ < param_.max_iter()) {
		Dtype loss = net_->ForwardBackward(bottom_vec);
		ComputeUpdateValue();
		net_->Update();

		pic_counts += Caffe::mutable_name2id().size();
		pos_triplets += Caffe::mutable_pos_triplets();
		triplets_count += Caffe::mutable_triplets().size();
		if (param_.display() && iter_ % param_.display() == 0) {
			gettimeofday(&finish_t, NULL);
			long int time_cost = (finish_t.tv_sec - tmp_t.tv_sec) * 1000000
					+ (finish_t.tv_usec - tmp_t.tv_usec);
			LOG(INFO)<< "Iteration " << iter_ << ", loss = " << loss
			<< ", image counts: " << (pic_counts * 1.0 / param_.display())
			<< ", triplets count: " << (triplets_count * 1.0 / param_.display())
			<< ", positive triplet: " << (pos_triplets * 1.0 / param_.display())
			<< ", cost time = " << (time_cost / 1000.0) << "ms";

			gettimeofday(&tmp_t, NULL);
			pic_counts = 0;
			pos_triplets = 0;
			triplets_count = 0;
		}
		if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
			// We need to set phase to test before running.
			Caffe::set_phase(Caffe::TEST);
			Test();
			Caffe::set_phase(Caffe::TRAIN);
		}
		// Check if we need to do snapshot
		if (param_.snapshot() && iter_ % param_.snapshot() == 0) {
			Snapshot();
		}
	}
	// After the optimization is done, always do a snapshot.
	iter_--;
	Snapshot();
	LOG(INFO)<< "Optimization Done.";
}

template<typename Dtype>
void Solver<Dtype>::Test() {
	LOG(INFO)<< "Iteration " << iter_ << ", Testing net";
	NetParameter net_param;
	net_->ToProto(&net_param);
	CHECK_NOTNULL(test_net_.get())->CopyTrainedLayersFrom(net_param);
	vector<Dtype> test_score;
	vector<Blob<Dtype>*> bottom_vec;
	for (int i = 0; i < param_.test_iter(); ++i) {
		const vector<Blob<Dtype>*>& result =
		test_net_->Forward(bottom_vec);
		if (i == 0) {
			for (int j = 0; j < result.size(); ++j) {
				const Dtype* result_vec = result[j]->cpu_data();
				for (int k = 0; k < result[j]->count(); ++k) {
					test_score.push_back(result_vec[k]);
				}
			}
		} else {
			int idx = 0;
			for (int j = 0; j < result.size(); ++j) {
				const Dtype* result_vec = result[j]->cpu_data();
				for (int k = 0; k < result[j]->count(); ++k) {
					test_score[idx++] += result_vec[k];
				}
			}
		}
	}
	for (int i = 0; i < test_score.size(); ++i) {
		LOG(INFO) << "Test score #" << i << ": "
		<< test_score[i] / param_.test_iter();
	}
}

template<typename Dtype>
void Solver<Dtype>::Snapshot() {
	NetParameter net_param;
	// For intermediate results, we will also dump the gradient values.
	net_->ToProto(&net_param, param_.snapshot_diff());
	string filename(param_.snapshot_prefix());
	char iter_str_buffer[20];
	sprintf(iter_str_buffer, "_iter_%d", iter_);
	filename += iter_str_buffer;
	LOG(INFO)<< "Snapshotting to " << filename;
	WriteProtoToBinaryFile(net_param, filename.c_str());
	SolverState state;
	SnapshotSolverState(&state);
	state.set_iter(iter_);
	state.set_learned_net(filename);
	filename += ".solverstate";
	LOG(INFO)<< "Snapshotting solver state to " << filename;
	WriteProtoToBinaryFile(state, filename.c_str());
}

template<typename Dtype>
void Solver<Dtype>::Restore(const char* state_file) {
	SolverState state;
	NetParameter net_param;
	ReadProtoFromBinaryFile(state_file, &state);
	if (state.has_learned_net()) {
		ReadProtoFromBinaryFile(state.learned_net().c_str(), &net_param);
		net_->CopyTrainedLayersFrom(net_param);
	}
	iter_ = state.iter();
	RestoreSolverState(state);
}

INSTANTIATE_CLASS(Solver);

}  // namespace caffe
