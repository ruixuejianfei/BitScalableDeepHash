// Copyright 2013 Yangqing Jia

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/lexical_cast.hpp>

#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>

#include <stdio.h>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <fstream>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/Util.hpp"
#include "caffe/vision_layers.hpp"

using std::string;
using std::vector;

#define PIC_LEN 256
#define SAME_CLASS_VAL 1
#define DIFF_CLASS_VAL 0.001

namespace caffe {
int cropsize_h = 0;
int cropsize_w = 0;
template<typename Dtype>
void getImgData(const int& id, const string& rootfolder, const string& filename,
		const int cropsize, const int channels, const int height,
		const int width, const bool crop_center, const bool mirror,
		Dtype* top_data, const Dtype* mean, const Dtype scale, const int size) {
	cv::Mat img = cv::imread(rootfolder + "/" + filename, CV_LOAD_IMAGE_COLOR);

	int h_off = 0, w_off = 0;
	int perturb = 18;
	if (cropsize) {

		// We only do random crop when we do training.
		if (Caffe::phase() == Caffe::TRAIN && true/*!crop_center*/) {
			h_off = rand() % (height - cropsize_h);
			w_off = rand() % (width - cropsize_w);
			h_off = (height - cropsize_h) / 2 + perturb / 2 - rand() % perturb;
			w_off = (width - cropsize_w) / 2 + perturb / 2 - rand() % perturb;
		} else {
			h_off = (height - cropsize_h) / 2;
			w_off = (width - cropsize_w) / 2;
		}
		bool mirror2 = true;
		if (Caffe::phase() == Caffe::TRAIN && mirror2 && rand() % 2) {
			// Copy mirrored version
			for (int c = 0; c < channels; ++c) {
				for (int h = 0; h < cropsize_h; ++h) {
					for (int w = 0; w < cropsize_w; ++w) {
						top_data[((id * channels + c) * cropsize_h + h)
								* cropsize_w + cropsize_w - 1 - w] = 128 * scale
								- (static_cast<uint8_t>(img.at < cv::Vec3b
										> (h + h_off, w + w_off)[c])
										- mean[(c * height + h + h_off) * width
												+ w + w_off]) * scale;
					}
				}
			}
		} else {
			// Normal copy
			for (int c = 0; c < channels; ++c) {
				for (int h = 0; h < cropsize_h; ++h) {
					for (int w = 0; w < cropsize_w; ++w) {
						top_data[((id * channels + c) * cropsize_h + h)
								* cropsize_w + w] = 128 * scale
								- (static_cast<uint8_t>(img.at < cv::Vec3b
										> (h + h_off, w + w_off)[c])
										- mean[(c * height + h + h_off) * width
												+ w + w_off]) * scale;
					}
				}
			}
		}
	} else {
		// we will prefer to use data() first, and then try float_data()

		for (int c = 0; c < channels; ++c) {
			for (int h = 0; h < img.rows; ++h) {
				for (int w = 0; w < img.cols; ++w) {
					top_data[((id * channels + c) * cropsize + h) * cropsize + w] =
							128 * scale
									- (static_cast<uint8_t>(img.at < cv::Vec3b
											> (h + h_off, w + w_off)[c])
											- mean[(c * height + h + h_off)
													* width + w + w_off])
											* scale;
				}
			}
		}
	}
}
template<typename Dtype>
Dtype* getFea(string pathName) {
	return NULL;
}

template<typename Dtype>
void addIfNotExist(std::map<string, int>& name2id, const string& rootfolder,
		const vector<string>& filenames, vector<int>& ids, const int cropsize,
		const int channels, const int height, const int width,
		const bool crop_center, const bool mirror, Dtype* top_data,
		const Dtype* mean, const Dtype scale, const int size,
		vector<int>& class_labels, std::vector<int>& imgclass) {
	for (int file_id = 0; file_id < filenames.size(); file_id++) {
		std::map<string, int>::iterator name_iter = name2id.find(
				filenames[file_id]);
		if (name_iter == name2id.end()) {
			//printf("\nbegin to load:%s%s",filenames[file_id].c_str() , ".jpg");
			//LOG(INFO)<<("\nbegin to load");
			getImgData(name2id.size(), rootfolder, filenames[file_id] + ".png",
					cropsize, channels, height, width, crop_center, mirror,
					top_data, mean, scale, size);
			//LOG(INFO)<<("\n load suc");
			int newid = name2id.size();
			name2id.insert(name_iter,
					std::make_pair(filenames[file_id], newid));
			imgclass.push_back(class_labels[file_id]);
			ids[file_id] = name2id.size() - 1;

		} else {
			ids[file_id] = name_iter->second;
		}
	}
}

template<typename Dtype>
void* DataLayerPrefetch(void* layer_pointer) {

	CHECK(layer_pointer);
	DataLayer<Dtype>* layer = reinterpret_cast<DataLayer<Dtype>*>(layer_pointer);
	CHECK(layer);

	if (Caffe::phase() == Caffe::TRAIN && layer->batch_iter_ != 0) {
//		LOG(INFO) << "Reusing previous batch";
		layer->batch_iter_--;
		return (void*) NULL;
	}
//	LOG(INFO) << "Generating new batch";
	layer->batch_iter_ = layer->layer_param_.batch_iter() - 1;

	Datum datum;
	CHECK(layer->prefetch_data_);
	CHECK(layer->prefetch_W_);
	Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
//	Dtype* top_label = layer->prefetch_label_->mutable_cpu_data();
	const Dtype scale = layer->layer_param_.scale();
	const int batchsize = layer->layer_param_.batchsize();
	const int cropsize = layer->layer_param_.cropsize();
	const bool mirror = layer->layer_param_.mirror();
	const int class_per_iter = layer->layer_param_.class_per_iter();
	const int triplet_per_class = layer->layer_param_.triplet_per_class();
	const int img_counts_per_class_per_iter =
			layer->img_counts_per_class_per_iter_;

	if (mirror && cropsize == 0) {
		LOG(FATAL)
				<< "Current implementation requires mirror and cropsize to be "
				<< "set at the same time.";
	}
	// datum scales
	const int channels = layer->datum_channels_;
	const int height = layer->datum_height_;
	const int width = layer->datum_width_;
	const int size = layer->datum_size_;
	const Dtype* mean = layer->data_mean_.cpu_data();

//	const std::map<string, int>& class2id = layer->class2id_;
	// 测试的时候不需要生成triplet
	if (Caffe::phase() == Caffe::TEST) {
		for (int i = 0;
				i < batchsize && layer->curIndex < layer->filenames_.size();
				i++, layer->curIndex++) {
			getImgData(i, layer->layer_param_.source(),
					layer->filenames_[layer->curIndex], cropsize, channels,
					height, width, layer->layer_param_.crop_center(), mirror,
					top_data, mean, scale, size);
		}
		return (void*) NULL;
	}

	const vector<string>& class_names = layer->class_names_;
	const vector<int>& img_counts_per_class = layer->img_counts_per_class_;
	vector < vector<string> > &triplets = Caffe::mutable_prefetch_triplets(); //Defined in common.hpp
	vector < vector<int> > &triplets_id = Caffe::mutable_prefetch_triplets_id();
	std::map<string, int>& name2id = Caffe::mutable_prefetch_name2id();
	std::vector<int>& imgclass = Caffe::mutable_prefetch_imgclass();
	name2id.clear();

	int img_total = class_per_iter * img_counts_per_class_per_iter;

	// 先选出class_per_iter那么多类
	vector<int> candidate_classes(class_names.size(), 0);
	for (int i = 0; i < candidate_classes.size(); i++) {
		candidate_classes[i] = i;
	}
	std::srand(unsigned(std::time(0)));
//	std::random_shuffle(selected_index.begin(), selected_index.end());
	random_unique(candidate_classes.begin(), candidate_classes.end(),
			MIN(candidate_classes.size(), class_per_iter));
	// 接着每类随机选img_counts_per_class_per_iter张图片
	vector < vector<int> > candidate_imgs(class_per_iter, vector<int>());
	for (int i = 0; i < class_per_iter; i++) {
		vector<int> tmp_indexs(
				img_counts_per_class[candidate_classes[i
						% candidate_classes.size()]], 0);
		for (int j = 0; j < tmp_indexs.size(); j++) {
			tmp_indexs[j] = j;
		}
		random_unique(tmp_indexs.begin(), tmp_indexs.end(),
				MIN(tmp_indexs.size(), img_counts_per_class_per_iter));
		for (int j = 0; j < img_counts_per_class_per_iter; j++) {
			candidate_imgs[i].push_back(
					tmp_indexs[j % tmp_indexs.size()] + layer->beg_index_);
		}
	}

	//LOG(INFO)<<"beg_index:"<<layer->beg_index_;
	memset(top_data, 0, sizeof(Dtype) * layer->prefetch_data_->count());
	std::vector<int> class_labels;
	for (int index_i = 0; index_i < class_per_iter; index_i++) {
		int class_index = candidate_classes[index_i % candidate_classes.size()];
		int img_counts = img_counts_per_class[class_index];
		class_labels.clear();
		class_labels.push_back(class_index);
		class_labels.push_back(class_index);
		for (int triplet_i = 0; triplet_i < triplet_per_class; triplet_i++) {
			int rand1 = rand() % candidate_imgs[index_i].size();
			int rand2 = rand() % candidate_imgs[index_i].size();
			if (rand2 == rand1) {
				rand2 = (rand1 + 1) % candidate_imgs[index_i].size();
			}
			int o1_index = candidate_imgs[index_i][rand1];
			int o2_index = candidate_imgs[index_i][rand2];
			int o3_tmp_index = (rand() % class_per_iter)
					% candidate_classes.size();
			int o3_class_index = candidate_classes[o3_tmp_index];
			if (o3_class_index == class_index) {
				o3_tmp_index = ((o3_tmp_index + 1) % class_per_iter)
						% candidate_classes.size();
				o3_class_index = candidate_classes[o3_tmp_index];
			}
			int o3_index = candidate_imgs[o3_tmp_index][rand()
					% candidate_imgs[o3_tmp_index].size()];

			string o1 = class_names[class_index] + "_" + boost::lexical_cast
					< string > (o1_index);
			string o2 = class_names[class_index] + "_" + boost::lexical_cast
					< string > (o2_index);
			string o3 = class_names[o3_class_index] + "_" + boost::lexical_cast
					< string > (o3_index);

			class_labels.push_back(o3_class_index);
			triplets[index_i * triplet_per_class + triplet_i][0] = o1;
			triplets[index_i * triplet_per_class + triplet_i][1] = o2;
			triplets[index_i * triplet_per_class + triplet_i][2] = o3;

			//LOG(INFO)<<("\nbegin to load triplet data");
			addIfNotExist(name2id, layer->layer_param_.source(),
					triplets[index_i * triplet_per_class + triplet_i],
					triplets_id[index_i * triplet_per_class + triplet_i],
					cropsize, channels, height, width,
					layer->layer_param_.crop_center(), mirror, top_data, mean,
					scale, size, class_labels, imgclass);
			//LOG(INFO)<<("\no problem");
			class_labels.pop_back();
		}
	}

	// LOG(INFO) << "Triplet Generated, computing L";

	// Calculate W matrix(see vision_layers.hpp)
	int img_num = imgclass.size();

	Dtype* w_data = layer->prefetch_W_->mutable_cpu_data();
	Dtype sums[img_total + 10];
	memset(w_data, 0, sizeof(Dtype) * img_total * img_total);
	memset(sums, 0, sizeof(Dtype) * img_total);
	for (int i = 0; i < img_total; i++) {
		for (int j = i; j < img_total; j++) {
			if (imgclass[i] == imgclass[j]) {
				w_data[i * img_total + j] = -SAME_CLASS_VAL;
				w_data[j * img_total + i] = -SAME_CLASS_VAL;
				sums[i] -= SAME_CLASS_VAL;
				sums[j] -= SAME_CLASS_VAL;
			} else {
				w_data[i * img_total + j] = -DIFF_CLASS_VAL;
				w_data[j * img_total + i] = -DIFF_CLASS_VAL;
				sums[i] -= DIFF_CLASS_VAL;
				sums[j] -= DIFF_CLASS_VAL;
			}
		}
	}

	for (int i = 0; i < img_total; i++) {
		w_data[i * img_total + i] += sums[i];

	}

	return (void*) NULL;
}

template<typename Dtype>
DataLayer<Dtype>::~DataLayer<Dtype>() {
// Finally, join the thread
	CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
}

template<typename Dtype>
void DataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	CHECK_EQ(bottom.size(), 0) << "Data Layer takes no input blobs.";
//	CHECK_EQ(top->size(), 2) << "Data Layer takes two blobs as output.";
	CHECK_EQ(top->size(), 2) << "Data Layer takes two blobs as output.";

	CHECK(this->layer_param_.has_img_files()) << "the file contain all the "
			<< "image names should be specified via attribute 'img_files' in prototxt";

	std::ifstream img_files_if(this->layer_param_.img_files().c_str());
	CHECK(img_files_if) << "Failed to open " << this->layer_param_.img_files();
	string line;
	counts_ = 0;
	std::map<std::string, int>::iterator name_iter_tmp;
	int name_id_tmp;
	class2id_.clear();
	class_names_.clear();
	img_counts_per_class_.clear();

	while (img_files_if >> line) {
		if (line.empty()) {
			continue;
		}
		counts_++;

		if (Caffe::phase() == Caffe::TEST) {
			counts_--;
			// 根据id的上下限筛选图片
			int index1 = line.find_last_of('_') + 1;
			int index2 = line.find_last_of('.');
			int id = atoi(line.substr(index1, index2 - index1).c_str());
			if (this->layer_param_.has_id_upper_bound()
					&& id > this->layer_param_.id_upper_bound())
				continue;
			if (this->layer_param_.has_id_lower_bound()
					&& id < this->layer_param_.id_lower_bound())
				continue;

			counts_++;
			filenames_.push_back(line);
//			LOG(INFO) << line;
			continue;
		}

		string class_name_tmp = line.substr(0, line.find_last_of('_'));
		name_iter_tmp = class2id_.find(class_name_tmp);
		if (name_iter_tmp == class2id_.end()) {
			name_id_tmp = class2id_.size();
			class2id_.insert(name_iter_tmp,
					std::make_pair(class_name_tmp, name_id_tmp));
			class_names_.push_back(class_name_tmp);
			img_counts_per_class_.push_back(1);
		} else {
			name_id_tmp = name_iter_tmp->second;
			img_counts_per_class_[name_id_tmp]++;
		}
	}
	img_files_if.close();

	int total_img_count = counts_;
	if (Caffe::phase() == Caffe::TRAIN) {
		batch_iter_ = 0;
		beg_index_ = 1;

		// 限制选择图片的数量
		if (this->layer_param_.has_id_upper_bound()) {
			total_img_count = 0;
			for (int i = 0; i < img_counts_per_class_.size(); i++) {
				img_counts_per_class_[i] = MIN(
						this->layer_param_.id_upper_bound(),
						img_counts_per_class_[i]);
				total_img_count += img_counts_per_class_[i];
			}
		}

		if (this->layer_param_.has_id_lower_bound()) {
			beg_index_ = this->layer_param_.id_lower_bound();
			total_img_count = 0;
			for (int i = 0; i < img_counts_per_class_.size(); i++) {
				img_counts_per_class_[i] = img_counts_per_class_[i] - beg_index_
						+ 1;
				CHECK_GT(img_counts_per_class_[i], 0);
				total_img_count += img_counts_per_class_[i];
				LOG(INFO) << class_names_[i] << ", img counts: "
						<< img_counts_per_class_[i] << "from " << beg_index_
						<< " to "
						<< (beg_index_ + img_counts_per_class_[i]) - 1;
			}
		}

	} else {
		curIndex = 0;
	}

	cv::Mat img;
	if (Caffe::phase() == Caffe::TRAIN) {
		img = cv::imread(
				this->layer_param_.source() + "/" + class_names_[0] + "_1.png",
				CV_LOAD_IMAGE_COLOR);
		LOG(INFO) << "begin to load img"
				<< this->layer_param_.source() + "/" + class_names_[0]
						+ "_1.png";
	} else {
		img = cv::imread(this->layer_param_.source() + "/" + filenames_[0],
				CV_LOAD_IMAGE_COLOR);
	}
// image
	int cropsize = this->layer_param_.cropsize();
	if (this->layer_param_.has_cropsize_h()) {
		cropsize_h = this->layer_param_.cropsize_h();
	} else
		cropsize_h = cropsize;
	if (this->layer_param_.has_cropsize_w()) {
		cropsize_w = this->layer_param_.cropsize_w();
	} else
		cropsize_w = cropsize;
// 测试的时候，不需要生成triplet，只需要计算所有图片的特征
	int batchsize = this->layer_param_.batchsize();
	if (Caffe::phase() == Caffe::TRAIN) {
		batchsize = this->layer_param_.class_per_iter()
				* this->layer_param_.triplet_per_class();

		Caffe::mutable_prefetch_triplets() = vector<vector<string> >(batchsize,
				vector<string>(3, string()));
		Caffe::mutable_prefetch_triplets_id() = vector<vector<int> >(batchsize,
				vector<int>(3, 0));

		batchsize *= 3;
		LOG(INFO) << "total_img_count: " << total_img_count << ", batchsize: "
				<< batchsize;
		batchsize = MIN(total_img_count, batchsize);
		if (this->layer_param_.has_id_upper_bound()) {
			int img_counts_bound = this->layer_param_.id_upper_bound()
					* this->layer_param_.class_per_iter();
			if (this->layer_param_.has_id_lower_bound()) {
				img_counts_bound -= this->layer_param_.id_lower_bound()
						* this->layer_param_.class_per_iter();
				img_counts_bound += this->layer_param_.class_per_iter();
			}
			LOG(INFO) << "img_counst_bound: " << img_counts_bound;
			batchsize = MIN(img_counts_bound, batchsize);
		}
		img_counts_per_class_per_iter_ =
				this->layer_param_.img_counts_per_class_per_iter();
		if (img_counts_per_class_per_iter_ == 0) {
			img_counts_per_class_per_iter_ = std::numeric_limits<int>::max();
		} else {
			batchsize = MIN(
					img_counts_per_class_per_iter_
							* this->layer_param_.class_per_iter(), batchsize);
		}
	}
	LOG(INFO) << "Max Batchsize: " << batchsize;

	if (cropsize > 0) {
		(*top)[0]->Reshape(batchsize, 3, cropsize_h, cropsize_w);
		prefetch_data_.reset(
				new Blob<Dtype>(batchsize, 3, cropsize_h, cropsize_w));
	} else {
		(*top)[0]->Reshape(batchsize, 3, img.rows, img.cols);
		prefetch_data_.reset(new Blob<Dtype>(batchsize, 3, img.rows, img.cols));
	}

	prefetch_W_.reset(new Blob<Dtype>(1, 1, batchsize, batchsize));

	LOG(INFO) << "l_matrix size: " << batchsize << ", " << batchsize;
	(*top)[1]->Reshape(1, 1, batchsize, batchsize);

	LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
			<< (*top)[0]->channels() << "," << (*top)[0]->height() << ","
			<< (*top)[0]->width();

	datum_channels_ = 3;
	datum_height_ = img.rows;
	datum_width_ = img.cols;
	datum_size_ = 3 * img.rows * img.cols;

	LOG(INFO) << datum_height_ << " " << cropsize << "!!";

	CHECK_GE(datum_height_, cropsize);
	CHECK_GE(datum_width_, cropsize);
// check if we want to have mean
	if (this->layer_param_.has_meanfile()) {
		BlobProto blob_proto;
		LOG(INFO) << "Loading mean file from" << this->layer_param_.meanfile();
		ReadProtoFromBinaryFile(this->layer_param_.meanfile().c_str(),
				&blob_proto);
		data_mean_.FromProto(blob_proto);
		CHECK_EQ(data_mean_.num(), 1);
		CHECK_EQ(data_mean_.channels(), datum_channels_);
		CHECK_EQ(data_mean_.height(), datum_height_);
		CHECK_EQ(data_mean_.width(), datum_width_);
	} else {
// Simply initialize an all-empty mean.
		data_mean_.Reshape(1, datum_channels_, datum_height_, datum_width_);
	}
// Now, start the prefetch thread. Before calling prefetch, we make two
// cpu_data calls so that the prefetch thread does not accidentally make
// simultaneous cudaMalloc calls when the main thread is running. In some
// GPUs this seems to cause failures if we do not so.
	prefetch_data_->mutable_cpu_data();
	prefetch_W_->mutable_cpu_data();
//	prefetch_label_->mutable_cpu_data();
	data_mean_.cpu_data();
	LOG(INFO) << "Initializing prefetch";
	CHECK(
			!pthread_create(&thread_, NULL, DataLayerPrefetch<Dtype>,
					reinterpret_cast<void*>(this)))
			<< "Pthread execution failed.";
	LOG(INFO) << "Prefetch initialized.";

// output info
//    LOG(INFO) << "Total class: " << class_names_.size() << ", Image counts per class as follow: ";
//    for (int i = 0; i < img_counts_per_class_.size(); i++) {
//        LOG(INFO) << class_names_[i] << ": " << img_counts_per_class_[i];
//	}
}

template<typename Dtype>
void DataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	// LOG(INFO) << "Passing data to next layer in CPU mode";
	// LOG(INFO) << "Joining prefetch thread: " << thread_;
	CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
	// CHECK((*top)[1]) << "L-Matrix Blob error";

	// LOG(INFO) << "l_matrix layer: " << (*top)[1]->height() << ' ' << (*top)[1]->width();
// Copy the data
	memcpy((*top)[0]->mutable_cpu_data(), prefetch_data_->cpu_data(),
			sizeof(Dtype) * prefetch_data_->count());

	memcpy((*top)[1]->mutable_cpu_data(), prefetch_W_->cpu_data(),
			sizeof(Dtype) * prefetch_W_->count());

	// LOG(INFO) << "Data passed to upper layers";

	Caffe::mutable_triplets() = Caffe::mutable_prefetch_triplets();
	Caffe::mutable_triplets_id() = Caffe::mutable_prefetch_triplets_id();
	Caffe::mutable_name2id() = Caffe::mutable_prefetch_name2id();

// Start a new prefetch thread
	CHECK(
			!pthread_create(&thread_, NULL, DataLayerPrefetch<Dtype>,
					reinterpret_cast<void*>(this)))
			<< "Pthread execution failed.";
}

template<typename Dtype>
void DataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top) {
	// LOG(INFO) << "Passing data to next layer in GPU mode";
// First, join the thread

	CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
//	 DataLayerPrefetch<Dtype>(this);
// Copy the data
	// LOG(INFO) << "Joining prefetch thread: " << thread_;
	// CHECK((*top)[1]) << "L-Matrix Blob error";

	// LOG(INFO) << "l_matrix layer: " << (*top)[1]->height() << ' ' << (*top)[1]->width();
	CUDA_CHECK(
			cudaMemcpy((*top)[0]->mutable_gpu_data(),
					prefetch_data_->cpu_data(),
					sizeof(Dtype) * prefetch_data_->count(),
					cudaMemcpyHostToDevice));
	CUDA_CHECK(
			cudaMemcpy((*top)[1]->mutable_gpu_data(), prefetch_W_->cpu_data(),
					sizeof(Dtype) * prefetch_W_->count(),
					cudaMemcpyHostToDevice));

	// BlobProto proto;
	// prefetch_W_->ToProto(&proto);
	// WriteProtoToBinaryFile(proto, "w_matrix.p");

	Caffe::mutable_triplets() = Caffe::mutable_prefetch_triplets();
	Caffe::mutable_triplets_id() = Caffe::mutable_prefetch_triplets_id();
	Caffe::mutable_name2id() = Caffe::mutable_prefetch_name2id();

	CHECK(
			!pthread_create(&thread_, NULL, DataLayerPrefetch<Dtype>,
					reinterpret_cast<void*>(this)))
			<< "Pthread execution failed.";
}

// The backward operations are dummy - they do not carry any computation.
template<typename Dtype>
Dtype DataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
	return Dtype(0.);
}

template<typename Dtype>
Dtype DataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
	return Dtype(0.);
}

INSTANTIATE_CLASS (DataLayer);

}  // namespace caffe
