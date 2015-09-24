// Copyright Yangqing Jia 2013

#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>

#include <stdint.h>
#include <fcntl.h>
#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <sys/stat.h>

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

using std::fstream;
using std::ios;
using std::max;
using std::string;
using std::string;
using std::vector;

using ::google::protobuf::Message;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;

namespace caffe {

void ReadProtoFromTextFile(const char* filename, Message* proto);
inline void ReadProtoFromTextFile(const string& filename, Message* proto) {
	ReadProtoFromTextFile(filename.c_str(), proto);
}

void WriteProtoToTextFile(const Message& proto, const char* filename);
inline void WriteProtoToTextFile(const Message& proto, const string& filename) {
	WriteProtoToTextFile(proto, filename.c_str());
}

void ReadProtoFromBinaryFile(const char* filename, Message* proto);
inline void ReadProtoFromBinaryFile(const string& filename, Message* proto) {
	ReadProtoFromBinaryFile(filename.c_str(), proto);
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename);
inline void WriteProtoToBinaryFile(const Message& proto,
		const string& filename) {
	WriteProtoToBinaryFile(proto, filename.c_str());
}

template<class T>
inline bool ReadImageToDatum(const string& filename, const vector<T>& labels,
		const int height, const int width, Datum* datum) {
	cv::Mat cv_img;
	if (height > 0 && width > 0) {
		cv::Mat cv_img_origin = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
		cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
	} else {
		cv_img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	}
	if (!cv_img.data) {
		LOG(ERROR)<< "Could not open or find file " << filename;
		return false;
	}

	datum->set_channels(3);
	datum->set_height(cv_img.rows);
	datum->set_width(cv_img.cols);
	// zhu
	// set label to be multidim
	//	datum->set_label(label);
	datum->mutable_label()->Clear();
	for (int i = 0; i < labels.size(); i++) {
		datum->mutable_label()->Add(labels[i]);
	}

	datum->clear_data();
	datum->clear_float_data();
	string* datum_string = datum->mutable_data();
	for (int c = 0; c < 3; ++c) {
		for (int h = 0; h < cv_img.rows; ++h) {
			for (int w = 0; w < cv_img.cols; ++w) {
				datum_string->push_back(
						static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
			}
		}
	}
	return true;
}

template<class T>
inline bool ReadImageToDatum(const string& filename, const vector<T>& labels,
		Datum* datum) {
	return ReadImageToDatum(filename, labels, 0, 0, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label, const int height,
		const int width, Datum* datum) {
	return ReadImageToDatum(filename, vector<int>(1, label), height, width,
			datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
		Datum* datum) {
	return ReadImageToDatum(filename, label, 0, 0, datum);
}

int CreateDir(const char *sPathName, int beg = 1);

}  // namespace caffe


#endif   // CAFFE_UTIL_IO_H_
