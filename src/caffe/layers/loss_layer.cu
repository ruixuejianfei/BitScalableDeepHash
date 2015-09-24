// Copyright 2013 Yangqing Jia
#include <algorithm>
#include <cmath>
#include <cfloat>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using std::max;
int iter_num = 0;

namespace caffe {

const float kLOG_THRESHOLD = 1e-20;

template<typename Dtype>
void MultinomialLogisticLossLayer<Dtype>::SetUp(
		const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
	CHECK_EQ(bottom.size(), 2) << "Loss Layer takes two blobs as input.";
	CHECK_EQ(top->size(), 0) << "Loss Layer takes no output.";
	CHECK_EQ(bottom[0]->num(), bottom[1]->num())
			<< "The data and label should have the same number.";
	CHECK_EQ(bottom[1]->channels(), 1);
	CHECK_EQ(bottom[1]->height(), 1);
	CHECK_EQ(bottom[1]->width(), 1);
}
;

template<typename Dtype>
Dtype MultinomialLogisticLossLayer<Dtype>::Backward_cpu(
		const vector<Blob<Dtype>*>& top, const bool propagate_down,
		vector<Blob<Dtype>*>* bottom) {
	const Dtype* bottom_data = (*bottom)[0]->cpu_data();
	const Dtype* bottom_label = (*bottom)[1]->cpu_data();
	Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
	int num = (*bottom)[0]->num();
	int dim = (*bottom)[0]->count() / (*bottom)[0]->num();
	memset(bottom_diff, 0, sizeof(Dtype) * (*bottom)[0]->count());
	Dtype loss = 0;
	// zhu
	// calculate loss for each input sample
	this->losses_.clear();
	for (int i = 0; i < num; ++i) {
		int label = static_cast<int>(bottom_label[i]);
		Dtype prob = max(bottom_data[i * dim + label], kLOG_THRESHOLD);
//		loss -= log(prob);
		this->losses_.push_back(-log(prob));
		loss += this->losses_[i];
		bottom_diff[i * dim + label] = -1. / prob / num;
	}
	return loss / num;
}

// TODO: implement the GPU version for multinomial loss

template<typename Dtype>
void InfogainLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	CHECK_EQ(bottom.size(), 2) << "Loss Layer takes two blobs as input.";
	CHECK_EQ(top->size(), 0) << "Loss Layer takes no output.";
	CHECK_EQ(bottom[0]->num(), bottom[1]->num())
			<< "The data and label should have the same number.";
	CHECK_EQ(bottom[1]->channels(), 1);
	CHECK_EQ(bottom[1]->height(), 1);
	CHECK_EQ(bottom[1]->width(), 1);
	BlobProto blob_proto;
	ReadProtoFromBinaryFile(this->layer_param_.source(), &blob_proto);
	infogain_.FromProto(blob_proto);
	CHECK_EQ(infogain_.num(), 1);
	CHECK_EQ(infogain_.channels(), 1);
	CHECK_EQ(infogain_.height(), infogain_.width());
}
;

template<typename Dtype>
Dtype InfogainLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
	const Dtype* bottom_data = (*bottom)[0]->cpu_data();
	const Dtype* bottom_label = (*bottom)[1]->cpu_data();
	const Dtype* infogain_mat = infogain_.cpu_data();
	Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
	int num = (*bottom)[0]->num();
	int dim = (*bottom)[0]->count() / (*bottom)[0]->num();
	CHECK_EQ(infogain_.height(), dim);
	Dtype loss = 0;
	// zhu
	// calculate loss for each input sample
	this->losses_.clear();
	for (int i = 0; i < num; ++i) {
		int label = static_cast<int>(bottom_label[i]);
		for (int j = 0; j < dim; ++j) {
			Dtype prob = max(bottom_data[i * dim + j], kLOG_THRESHOLD);
//			loss -= infogain_mat[label * dim + j] * log(prob);
			this->losses_.push_back(-infogain_mat[label * dim + j] * log(prob));
			loss += this->losses_[i];
			bottom_diff[i * dim + j] = -infogain_mat[label * dim + j] / prob
					/ num;
		}
	}
	return loss / num;
}

template<typename Dtype>
void EuclideanLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	CHECK_EQ(bottom.size(), 2) << "Loss Layer takes two blobs as input.";
	CHECK_EQ(top->size(), 0) << "Loss Layer takes no as output.";
	CHECK_EQ(bottom[0]->num(), bottom[1]->num())
			<< "The data and label should have the same number.";
	CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
	CHECK_EQ(bottom[0]->height(), bottom[1]->height());
	CHECK_EQ(bottom[0]->width(), bottom[1]->width());
	difference_.Reshape(bottom[0]->num(), bottom[0]->channels(),
			bottom[0]->height(), bottom[0]->width());
}

template<typename Dtype>
Dtype EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
	int count = (*bottom)[0]->count();
	int num = (*bottom)[0]->num();
	caffe_sub(count, (*bottom)[0]->cpu_data(), (*bottom)[1]->cpu_data(),
			difference_.mutable_cpu_data());

//	Dtype loss = caffe_cpu_dot(count, difference_.cpu_data(),
//			difference_.cpu_data()) / num / Dtype(2);
	// zhu
	// calculate loss for each input sample
	Dtype loss = 0;
	this->losses_.clear();
	int dim = count / num;
	for (int i = 0; i < num; i++) {
		this->losses_.push_back(
				caffe_cpu_dot(dim, difference_.cpu_data(),
						difference_.cpu_data()));
		loss += this->losses_[i];
	}
	loss = loss / num / Dtype(2);

	// Compute the gradient
	caffe_axpby(count, Dtype(1) / num, difference_.cpu_data(), Dtype(0),
			(*bottom)[0]->mutable_cpu_diff());
	return loss;
}

template<typename Dtype>
void EuclideanTripletLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	CHECK_EQ(bottom.size(), 2) << "Loss Layer takes two blobs as input.";
	CHECK_EQ(top->size(), 0) << "Loss Layer takes no as output.";
//	CHECK_EQ(bottom[0]->num(), bottom[1]->num())
//	<< "The data and label should have the same number.";
//	CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
//	CHECK_EQ(bottom[0]->height(), bottom[1]->height());
//	CHECK_EQ(bottom[0]->width(), bottom[1]->width());
	LOG(INFO) << "begin to init loss layer" << bottom[0]->num() << " "
			<< bottom[0]->channels() << " " << bottom[0]->height()
			<< bottom[0]->width();
	difference_.Reshape(bottom[0]->num(), bottom[0]->channels(),
			bottom[0]->height(), bottom[0]->width());
	s_.Reshape(bottom[0]->num(), 1, 1, 1);
	laplacian_beta_ = this->layer_param_.laplacian_beta();
	LOG(INFO) << "Laplacian_beta: " << laplacian_beta_;
	LOG(INFO) << "loss layer inited";

	loss_threshold_ = (
			this->layer_param_.has_loss_threshold() ?
					this->layer_param_.loss_threshold() : Dtype(-1.0));

	iter_decay = this->layer_param_.iter_decay();
	decay = this->layer_param_.decay();

}
//*****************************************NO NORM*********************************************//
bool to_norm_ = false;
template<typename Dtype>
Dtype EuclideanTripletLossLayer<Dtype>::Backward_cpu(
		const vector<Blob<Dtype>*>& top, const bool propagate_down,
		vector<Blob<Dtype>*>* bottom) {
	iter_num++;
	if (iter_num % iter_decay == 0)
		loss_threshold_ += decay;
	LOG(INFO) << "loss_threshold: " << loss_threshold_;

	int count = (*bottom)[0]->count();
	int num = (*bottom)[0]->num();
	int size = count / num;
	memset(difference_.mutable_cpu_data(), 0, sizeof(Dtype) * count);
	const std::vector<std::vector<int> >& triplets_id =
			Caffe::mutable_triplets_id();

	const std::vector<std::vector<std::string> >& triplets =
			Caffe::mutable_triplets();
	const std::map<std::string, int>& name2id = Caffe::mutable_name2id();
	Blob < Dtype > intermediate_result(1, size, 1, 1);

	Blob < Dtype
			> diff_bottom((*bottom)[0]->num(), (*bottom)[0]->channels(),
					(*bottom)[0]->height(), (*bottom)[0]->width());

	//normalize the output
	if (to_norm_ == true) {
		for (std::map<std::string, int>::const_iterator iter = name2id.begin();
				iter != name2id.end(); iter++) {
			int size = (*bottom)[0]->channels() * (*bottom)[0]->height()
					* (*bottom)[0]->width();
			Dtype* addr = (*bottom)[0]->mutable_cpu_data()
					+ iter->second * size;
			Dtype s = caffe_cpu_dot(size, addr, addr);
			s_.mutable_cpu_data()[iter->second] = s;
			//printf("/ns=%f",s);
			if (s != 0)
				for (int i = 0; i < size; i++)
					addr[i] = addr[i] / sqrt(s);
		}
	}
	//*********** Compute loss (Laplacian + Triplet) *********
	Dtype total_loss = 0;
	//***************** Compute Laplacian Loss *****************
	//LOG(INFO)<< (*bottom)[0]->count()<<" "<<(*bottom)[0]->num() << " " << (*bottom)[0]->channels()<<" " << (*bottom)[1]->height() <<" "<<(*bottom)[0]->width();
	//for (int i = 0; i < (*bottom)[1]->count();i++)
	//	LOG(INFO) << (*bottom)[1]->cpu_data()[i];
	Blob < Dtype > *v_blob = (*bottom)[0];
	Blob < Dtype > *l_blob = (*bottom)[1];

	cv::Mat v_mat(v_blob->channels(), v_blob->num(), CV_32F,
			v_blob->mutable_cpu_data());
	// cv::FileStorage fsv("v_mat.yml", FileStorage::WRITE);
	// fsv << "mat" << v_mat;
	// fsv.release();

	// LOG(INFO) << "v_mat: " << v_mat.rows << ", " <<  v_mat.cols;
	cv::Mat l_mat(l_blob->height(), l_blob->width(), CV_32F,
			l_blob->mutable_cpu_data());
	// cv::FileStorage fsl("l_mat.yml", FileStorage::WRITE);
	// fsl << "mat" << l_mat;
	// fsl.release();
	// LOG(INFO) << "l_mat: " << l_mat.rows << ", " <<  l_mat.cols;
	Dtype laplacian_loss = laplacian_beta_
			* cv::trace(v_mat * l_mat * v_mat.t())[0];
	total_loss += laplacian_loss;
	LOG(INFO) << "Laplacian loss: " << laplacian_loss;

	//****************** Compute Triplet Loss ******************
	vector < Dtype > loss_per_triplet(triplets_id.size(), 0);
	int& pos_triplets = Caffe::mutable_pos_triplets();
	pos_triplets = 0;
	for (int triplet_i = 0; triplet_i < triplets_id.size(); triplet_i++) {
		memset(intermediate_result.mutable_cpu_data(), 0, sizeof(Dtype) * size);

		const int o1_index = triplets_id[triplet_i][0];
		const int o2_index = triplets_id[triplet_i][1];
		const int o3_index = triplets_id[triplet_i][2];
		const Dtype* x1_addr = (*bottom)[0]->cpu_data() + o1_index * size;
		const Dtype* x2_addr = (*bottom)[0]->cpu_data() + o2_index * size;
		const Dtype* x3_addr = (*bottom)[0]->cpu_data() + o3_index * size;

		// x1 - x2
		caffe_sub(size, x1_addr, x2_addr,
				intermediate_result.mutable_cpu_data());
		loss_per_triplet[triplet_i] = caffe_cpu_dot(size,
				intermediate_result.cpu_data(), intermediate_result.cpu_data());

		// x1 - x3
		caffe_sub(size, x1_addr, x3_addr,
				intermediate_result.mutable_cpu_data());
		loss_per_triplet[triplet_i] -= caffe_cpu_dot(size,
				intermediate_result.cpu_data(), intermediate_result.cpu_data());

		bool addflage = false;

		if (addflage) {
			std::cout << "triplet id:" << triplet_i << ",";
			std::cout << triplets[triplet_i][0] << ", ";
			std::cout << triplets[triplet_i][1] << ", ";
			std::cout << triplets[triplet_i][2] << ": ";
			std::cout << triplet_i << ": " << loss_per_triplet[triplet_i]
					<< ", " << total_loss << std::endl;
		}

		total_loss += loss_per_triplet[triplet_i];
		if (loss_per_triplet[triplet_i] > 0)
			pos_triplets++;
	}

	Blob < Dtype > RL(num, size, 1, 1);
	Blob < Dtype > Lr(num, size, 1, 1);
	memset(RL.mutable_cpu_data(), 0, sizeof(Dtype) * count);
	memset(Lr.mutable_cpu_data(), 0, sizeof(Dtype) * count);
	for (std::map<std::string, int>::const_iterator iter = name2id.begin();
			iter != name2id.end(); iter++) {
		int size = (*bottom)[0]->channels() * (*bottom)[0]->height()
				* (*bottom)[0]->width();

		for (int triplet_i = 0; triplet_i < triplets_id.size(); triplet_i++) {
			if (loss_per_triplet[triplet_i] < loss_threshold_) {
				continue;
			}

			const int o1_index = triplets_id[triplet_i][0];
			const int o2_index = triplets_id[triplet_i][1];
			const int o3_index = triplets_id[triplet_i][2];
			const Dtype* x1_addr = (*bottom)[0]->cpu_data() + o1_index * size;
			const Dtype* x2_addr = (*bottom)[0]->cpu_data() + o2_index * size;
			const Dtype* x3_addr = (*bottom)[0]->cpu_data() + o3_index * size;
			const Dtype* addr1 = NULL;
			const Dtype* addr2 = NULL;
			if (iter->second == o1_index) {
				addr1 = x3_addr;
				addr2 = x2_addr;
			} else if (iter->second == o2_index) {
				addr1 = x2_addr;
				addr2 = x1_addr;
			} else if (iter->second == o3_index) {
				addr1 = x1_addr;
				addr2 = x3_addr;
			} else {
				continue;
			}

			memset(intermediate_result.mutable_cpu_data(), 0,
					sizeof(Dtype) * size);
			caffe_sub(size, addr1, addr2,
					intermediate_result.mutable_cpu_data());
			caffe_axpby(size, Dtype(2), intermediate_result.cpu_data(),
					Dtype(1),
					difference_.mutable_cpu_data() + iter->second * size);
		}

		Dtype* diff_addr = difference_.mutable_cpu_data() + iter->second * size;
		Dtype* diff_addr_bottom = diff_bottom.mutable_cpu_data()
				+ iter->second * size;
		Dtype* addr = (*bottom)[0]->mutable_cpu_data() + iter->second * size;
		Dtype s = s_.mutable_cpu_data()[iter->second];
		if (to_norm_ == true) {
			for (int i = 0; i < size; i++) {
				//(1/sqrt(s)-s*y_i^2*s^{-3/2}
				diff_addr_bottom[i] = 0;
				for (int j = 0; j < size; j++) {
					if (j == i)
						diff_addr_bottom[i] +=
								diff_addr[i]
										* (1 / sqrt(s)
												- s * addr[i] * addr[i]
														* (1 / sqrt(s))
														* (1 / sqrt(s))
														* (1 / sqrt(s)));
					else {

						diff_addr_bottom[i] -= diff_addr[j] * s * addr[i]
								* addr[j] * (1 / sqrt(s)) * (1 / sqrt(s))
								* (1 / sqrt(s));
					}
				}
			}
		}
		//************* Laplacian diff *************

		for (int m = 0; m < size; m++) {
			Dtype temp = 0;
			for (int n = 0; n < num; n++) {
				if (n == iter->second)
					continue;
				temp += (*bottom)[0]->mutable_cpu_data()[n * size + m]
						* (*bottom)[1]->mutable_cpu_data()[n * num
								+ iter->second];
			}
			RL.mutable_cpu_data()[iter->second * size + m] = temp;
			Lr.mutable_cpu_data()[iter->second * size + m] =
					(*bottom)[1]->mutable_cpu_data()[iter->second * num
							+ iter->second]
							* (*bottom)[0]->mutable_cpu_data()[iter->second
									* size + m];
		}
	}
	caffe_add(count, RL.cpu_data(), Lr.cpu_data(), RL.mutable_cpu_data());
	caffe_scal(count, (Dtype) laplacian_beta_, RL.mutable_cpu_data());
	caffe_add(count, RL.cpu_data(), difference_.cpu_data(),
			difference_.mutable_cpu_data());

	if (to_norm_ == true)
		caffe_axpby(count, Dtype(1) / triplets_id.size(),
				diff_bottom.cpu_data(), Dtype(0),
				(*bottom)[0]->mutable_cpu_diff());
	else
		caffe_axpby(count, Dtype(1) / triplets_id.size(),
				difference_.cpu_data(), Dtype(0),
				(*bottom)[0]->mutable_cpu_diff());

	return total_loss;
}

template<typename Dtype>
void AccuracyLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	CHECK_EQ(bottom.size(), 2) << "Accuracy Layer takes two blobs as input.";
	CHECK_EQ(top->size(), 1) << "Accuracy Layer takes 1 output.";
	CHECK_EQ(bottom[0]->num(), bottom[1]->num())
			<< "The data and label should have the same number.";
	CHECK_EQ(bottom[1]->channels(), 1);
	CHECK_EQ(bottom[1]->height(), 1);
	CHECK_EQ(bottom[1]->width(), 1);
	(*top)[0]->Reshape(1, 2, 1, 1);
}

template<typename Dtype>
void AccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	Dtype accuracy = 0;
	Dtype logprob = 0;
	const Dtype* bottom_data = bottom[0]->cpu_data();
	const Dtype* bottom_label = bottom[1]->cpu_data();
	int num = bottom[0]->num();
	int dim = bottom[0]->count() / bottom[0]->num();
	for (int i = 0; i < num; ++i) {
		// Accuracy
		Dtype maxval = -FLT_MAX;
		int max_id = 0;
		for (int j = 0; j < dim; ++j) {
			if (bottom_data[i * dim + j] > maxval) {
				maxval = bottom_data[i * dim + j];
				max_id = j;
			}
		}
		if (max_id == (int) bottom_label[i]) {
			++accuracy;
		}
		Dtype prob = max(bottom_data[i * dim + (int) bottom_label[i]],
				kLOG_THRESHOLD);
		logprob -= log(prob);
	}
	// LOG(INFO)  << "Accuracy: " << accuracy;
	(*top)[0]->mutable_cpu_data()[0] = accuracy / num;
	(*top)[0]->mutable_cpu_data()[1] = logprob / num;
}

INSTANTIATE_CLASS (MultinomialLogisticLossLayer);
INSTANTIATE_CLASS (InfogainLossLayer);
INSTANTIATE_CLASS (EuclideanLossLayer);
INSTANTIATE_CLASS (EuclideanTripletLossLayer);
INSTANTIATE_CLASS (AccuracyLayer);

}  // namespace caffe

