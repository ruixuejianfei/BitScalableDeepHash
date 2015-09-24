// Copyright 2013 Yangqing Jia
//
// This is a simple script that allows one to quickly test a network whose
// structure is specified by text format protocol buffers, and whose parameter
// are loaded from a pre-trained network.
// Usage:
//    test_net net_proto pretrained_net_proto iterations [CPU/GPU]

#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>
#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <termios.h>
#include <unistd.h>

#include "caffe/caffe.hpp"

using namespace caffe;
using namespace std;

void InsertSort(float *arr, int *ind, int n);

int main(int argc, char** argv) {
	if (argc < 3) {
		LOG(ERROR)
				<< "test_net net_proto pretrained_net_proto [output_feature_path] [CPU/GPU]";
		return 0;
	}

	cudaSetDevice(0);
	Caffe::set_phase(Caffe::TEST);
	LOG(INFO) << argv[0] << "\t" << argv[4];
	if (argc >= 6 && strcmp(argv[5], "CPU") == 0) {
		LOG(INFO) << "Using CPU";
		Caffe::set_mode(Caffe::CPU);
	} else {
		LOG(INFO) << "Using GPU";
		Caffe::set_mode(Caffe::GPU);
	}

	if (argc >= 7) {
		LOG(INFO) << "Using " << argv[6] << " bits hashing code.";
	}

	vector < shared_ptr<Blob<float> > > query_feature_map;
	vector < shared_ptr<Blob<float> > > database_feature_map;
	vector<int> query_img_count_per_class;
	vector<int> database_img_count_per_class;

	vector<int> query_class_ids;
	vector<int> database_class_ids;
	map<string, int> query_class2id;
	map<string, int> database_class2id;
	int query_data_counts = 0;
	int database_data_counts = 0;
	int FEA_SIZE = 0;
	NetParameter test_net_param_query;
	ReadProtoFromTextFile(argv[1], &test_net_param_query);
	Net<float> caffe_test_net_query(test_net_param_query);
	NetParameter test_net_param_database;
	ReadProtoFromTextFile(argv[2], &test_net_param_database);
	Net<float> caffe_test_net_database(test_net_param_database);
	NetParameter trained_net_param;
	ReadProtoFromBinaryFile(argv[3], &trained_net_param);
	caffe_test_net_query.CopyTrainedLayersFrom(trained_net_param);
	caffe_test_net_database.CopyTrainedLayersFrom(trained_net_param);

	//***************************** query ****************************
	vector < shared_ptr<Layer<float> > > layers = caffe_test_net_query.layers();

	//*********************** achieve weight ****************************
	ElementWiseProductLayer<float> *elewiselayer =
			dynamic_cast<ElementWiseProductLayer<float>*>(layers[13].get());
	CHECK(elewiselayer);

	vector < shared_ptr<Blob<float> > > &myblobs = elewiselayer->blobs();
	Blob<float> ori_weight(1, myblobs[0]->count(), 1, 1);
	memset(ori_weight.mutable_cpu_data(), 0,
			sizeof(float) * myblobs[0]->count());
	caffe_copy(myblobs[0]->count(), myblobs[0]->cpu_data(),
			ori_weight.mutable_cpu_data());
	float* weight = myblobs[0]->mutable_cpu_data();

	int nbits = atoi(argv[6]);
	if (myblobs[0]->count() < nbits) {
		LOG(ERROR) << "The number of bits set in test phrase (" << nbits
				<< ") is larger than in training phrase ("
				<< myblobs[0]->count() << ").";
		return 0;
	}

	int* index = new int[myblobs[0]->count()];
	for (int i = 0; i < myblobs[0]->count(); i++) {
		weight[i] = abs(weight[i]);
		index[i] = i;
	}

	for (int i = 0; i < myblobs[0]->count(); i++)
		LOG(INFO) << ori_weight.mutable_cpu_data()[i] << " " << weight[i]
				<< "  " << index[i];
	InsertSort(weight, index, myblobs[0]->count());

	const DataLayer<float> *datalayer =
			dynamic_cast<const DataLayer<float>*>(layers[0].get());
	const vector<string>& filenames = datalayer->getFilenames();
	{
		CHECK(datalayer);

		FEA_SIZE =
				(*(caffe_test_net_query.bottom_vecs().rbegin()))[0]->channels()
						* (*(caffe_test_net_query.bottom_vecs().rbegin()))[0]->width()
						* (*(caffe_test_net_query.bottom_vecs().rbegin()))[0]->height();
		LOG(INFO) << "FEA_SIZE: " << FEA_SIZE;

		query_data_counts = datalayer->getDataCount();

		vector<Blob<float>*> dummy_blob_input_vec;

		query_class_ids.resize(query_data_counts, 0);
		for (int i = 0; i < query_data_counts; i++) {
			string prefix = filenames[i].substr(0,
					filenames[i].find_last_of('_'));
			map<string, int>::iterator iter = query_class2id.find(prefix);
			if (iter != query_class2id.end()) {
				query_class_ids[i] = iter->second;
				query_img_count_per_class[iter->second]++;
			} else {
				query_class_ids[i] = query_class2id.size();
				query_class2id.insert(iter,
						make_pair(prefix, query_class2id.size()));
				query_img_count_per_class.push_back(1);
			}
		}

		for (int i = 0; i < query_data_counts; i++) {
			query_feature_map.push_back(
					shared_ptr < Blob<float>
							> (new Blob<float>(1, FEA_SIZE, 1, 1)));
		}

		int batchCount =
				std::ceil(
						query_data_counts
								/ (floor)(
										test_net_param_query.layers(0).layer().batchsize()));
		int batchsize = test_net_param_query.layers(0).layer().batchsize();
		for (int batch_id = 0, file_id = 0;
				batch_id < batchCount && file_id < query_data_counts;
				++batch_id) {
			LOG(INFO) << "Total batchs (query): " << batchCount
					<< ", Processing batch (query): " << (batch_id + 1);
			const vector<Blob<float>*>& result = caffe_test_net_query.Forward(
					dummy_blob_input_vec);

			Blob<float>* features;
			features = (*(caffe_test_net_query.bottom_vecs().begin() + 13))[0];

			for (int k = 0; file_id < query_data_counts && k < batchsize;
					file_id++, k++) {
				memcpy(query_feature_map[file_id]->mutable_cpu_data(),
						features->cpu_data() + k * FEA_SIZE,
						sizeof(float) * FEA_SIZE);
			}
		}
		if (argc > 4) {
			printf("\nbegin to write feature:%s,%s",
					(string(argv[4]) + "/query_filenames.txt").c_str(),
					(string(argv[4]) + "/query_features.txt").c_str());
			LOG(INFO) << "feature file:"
					<< (string(argv[4]) + "/query_filenames.txt").c_str();
			ofstream fNameFile(
					(string(argv[4]) + "/query_filenames.txt").c_str());
			ofstream feaFile((string(argv[4]) + "/query_features.txt").c_str());
			for (int file_id = 0; file_id < filenames.size(); file_id++) {
				fNameFile << filenames[file_id] << "\n";
				const float* feas = query_feature_map[file_id]->cpu_data();
				for (int fea_i = 0; fea_i < FEA_SIZE; fea_i++) {
					feaFile << feas[fea_i] << " ";
				}
				feaFile << "\n";
			}
			fNameFile.close();
			feaFile.close();
		}
	}

	//***************************** database ****************************
	vector < shared_ptr<Layer<float> > > layers_database =
			caffe_test_net_database.layers();
	const DataLayer<float> *datalayer_database = dynamic_cast<const DataLayer<
			float>*>(layers_database[0].get());
	const vector<string>& filenames_database =
			datalayer_database->getFilenames();
	{
		CHECK(datalayer_database);

		FEA_SIZE =
				(*(caffe_test_net_database.bottom_vecs().rbegin()))[0]->channels()
						* (*(caffe_test_net_database.bottom_vecs().rbegin()))[0]->width()
						* (*(caffe_test_net_database.bottom_vecs().rbegin()))[0]->height();
		LOG(INFO) << "FEA_SIZE: " << FEA_SIZE;

		database_data_counts = datalayer_database->getDataCount();

		vector<Blob<float>*> dummy_blob_input_vec;

		database_class_ids.resize(database_data_counts, 0);
		database_img_count_per_class.clear();
		database_class2id.clear();
		for (int i = 0; i < database_data_counts; i++) {
			string prefix = filenames_database[i].substr(0,
					filenames_database[i].find_last_of('_'));
			map<string, int>::iterator iter = database_class2id.find(prefix);
			if (iter != database_class2id.end()) {
				database_class_ids[i] = iter->second;
				database_img_count_per_class[iter->second]++;
			} else {
				database_class_ids[i] = database_class2id.size();
				database_class2id.insert(iter,
						make_pair(prefix, database_class2id.size()));
				database_img_count_per_class.push_back(1);
			}
		}

		for (int i = 0; i < database_data_counts; i++) {
			database_feature_map.push_back(
					shared_ptr < Blob<float>
							> (new Blob<float>(1, FEA_SIZE, 1, 1)));
		}

		int batchCount =
				std::ceil(
						database_data_counts
								/ (floor)(
										test_net_param_database.layers(0).layer().batchsize()));
		int batchsize = test_net_param_database.layers(0).layer().batchsize();
		for (int batch_id = 0, file_id = 0;
				batch_id < batchCount && file_id < database_data_counts;
				++batch_id) {
			LOG(INFO) << "Total batchs (database): " << batchCount
					<< ", Processing batch (database): " << (batch_id + 1);
			const vector<Blob<float>*>& result =
					caffe_test_net_database.Forward(dummy_blob_input_vec);

			Blob<float>* features;
			features =
					(*(caffe_test_net_database.bottom_vecs().begin() + 13))[0];

			for (int k = 0; file_id < database_data_counts && k < batchsize;
					file_id++, k++) {
				memcpy(database_feature_map[file_id]->mutable_cpu_data(),
						features->cpu_data() + k * FEA_SIZE,
						sizeof(float) * FEA_SIZE);
			}
		}
		if (argc > 5) {
			printf("\nbegin to write feature:%s,%s",
					(string(argv[4]) + "/database_filenames.txt").c_str(),
					(string(argv[4]) + "/database_features.txt").c_str());
			LOG(INFO) << "feature file:"
					<< (string(argv[4]) + "/database_filenames.txt").c_str();
			ofstream fNameFile(
					(string(argv[4]) + "/database_filenames.txt").c_str());
			ofstream feaFile(
					(string(argv[4]) + "/database_features.txt").c_str());
			for (int file_id = 0; file_id < filenames_database.size();
					file_id++) {
				fNameFile << filenames_database[file_id] << "\n";
				const float* feas = database_feature_map[file_id]->cpu_data();
				for (int fea_i = 0; fea_i < FEA_SIZE; fea_i++) {
					feaFile << feas[fea_i] << " ";
				}
				feaFile << "\n";
			}
			fNameFile.close();
			feaFile.close();
		}
	}

	streambuf* buf = cout.rdbuf();
	ostream out(buf);

	float accs[query_data_counts][30];
	float acc_per_class[query_class2id.size()][30];

	float ap[query_data_counts];
	float ave_pre[query_class2id.size()];
	float mean_ave_pre = 0;

	memset(ap, 0, sizeof(ap));
	memset(ave_pre, 0, sizeof(ave_pre));

	{
		Blob<float> intermediate_result(1, FEA_SIZE, 1, 1);

		for (int i = 0; i < query_data_counts; i++) {
			int i_class = query_class_ids[i];
			multimap<float, int> loss2id;
			float loss;
			for (int j = 0; j < database_data_counts; j++) {
				caffe_sub(FEA_SIZE, query_feature_map[i]->cpu_data(),
						database_feature_map[j]->cpu_data(),
						intermediate_result.mutable_cpu_data());
				/*loss = caffe_cpu_dot(FEA_SIZE, intermediate_result.cpu_data(),
				 intermediate_result.cpu_data());*/

				//******* ranking loss: -z_i*z_j
				/*caffe_mul(FEA_SIZE, query_feature_map[i]->cpu_data(),
				 database_feature_map[j]->cpu_data(),
				 intermediate_result.mutable_cpu_data());*/

				loss = 0;
				for (int w = 1; w <= nbits; w++) {
					int imp_w = FEA_SIZE - w;
					loss +=
							intermediate_result.mutable_cpu_data()[index[imp_w]]
									* intermediate_result.mutable_cpu_data()[index[imp_w]];

				}
				loss2id.insert(make_pair(loss, j));
			}
			multimap<float, int>::iterator iter = loss2id.begin();

			LOG(INFO) << "query: " << filenames[i];
			int right = 0;
			float ap_temp = 0;
			for (int k = 1; iter != loss2id.end(); k++, iter++) {
				if (i_class == database_class_ids[iter->second]) {
					right++;
					ap_temp = right * 1.0 / k;
					ap[i] = ap[i] + ap_temp;
				}
			}
			ave_pre[i_class] = ave_pre[i_class]
					+ ap[i] / database_img_count_per_class[i_class];
		}
	}

	cout << endl << query_class2id.size() << endl;
	for (int i = 0; i < query_class2id.size(); i++) {
		cout << i << "\t" << ave_pre[i] / query_img_count_per_class[i] << endl;
		mean_ave_pre = mean_ave_pre + ave_pre[i];
	}

	cout << "mean_ave_pre : " << (mean_ave_pre / query_data_counts) << " "
			<< mean_ave_pre << endl;
	return 0;
}

void InsertSort(float *arr, int *ind, int n) {
	for (int i = 1; i < n; i++) {
		if (arr[i] < arr[i - 1]) {
			int j = i - 1;
			float x = arr[i];
			int y = ind[i];
			arr[i] = arr[i - 1];
			ind[i] = ind[i - 1];
			while (x < arr[j]) {
				arr[j + 1] = arr[j];
				ind[j + 1] = ind[j];
				j--;
			}
			arr[j + 1] = x;
			ind[j + 1] = y;
		}
	}

}
