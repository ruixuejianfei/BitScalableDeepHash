// Copyright 2013 Yangqing Jia
//
// This is a simple script that allows one to quickly train a network whose
// parameters are specified by text format protocol buffers.
// Usage:
//    train_net net_proto_file solver_proto_file [resume_point_file]

#include <cuda_runtime.h>

#include <cstring>

#include "caffe/caffe.hpp"

using namespace caffe;

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
	if (argc < 3) {
		LOG(ERROR) << "Usage: \n" << "train_net_zhu solver_proto_file"
				<< "output_net_name [resume_point_file]";
		return 0;
	}

	SolverParameter solver_param;
	ReadProtoFromTextFile(argv[1], &solver_param);

	LOG(INFO) << "Starting Optimization";
	SGDSolver<float> solver(solver_param);

	if (argc == 4) {
		LOG(INFO) << "Resuming from " << argv[3];
		solver.Solve(argv[3]);
	} else {
		solver.Solve();
	}
	LOG(INFO) << "Optimization Done.";

	LOG(INFO) << "Saving Model";
	string output_net_name = argv[2];
	//	if (argc > 4)
	//		output_net_name = argv[3];
	//	else {
	//		output_net_name = argv[1];
	//		output_net_name.append("_net_proto");
	//	}
	NetParameter output_net_param;
	solver.net()->ToProto(&output_net_param, true);
	WriteProtoToBinaryFile(output_net_param, output_net_name);
	//	Net<float> net(solver_param.train_net());
	//	net.CopyTrainedLayersFrom(argv[2]);

	return 0;
}
