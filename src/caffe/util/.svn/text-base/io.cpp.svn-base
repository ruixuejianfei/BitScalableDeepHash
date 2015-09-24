// Copyright 2013 Yangqing Jia

#include "caffe/util/io.hpp"


namespace caffe {

void ReadProtoFromTextFile(const char* filename,
		::google::protobuf::Message* proto) {
	int fd = open(filename, O_RDONLY);
	CHECK_NE(fd, -1)<< "File not found: " << filename;
	FileInputStream* input = new FileInputStream(fd);
	CHECK(google::protobuf::TextFormat::Parse(input, proto));
	delete input;
	close(fd);
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
	int fd = open(filename, O_WRONLY);
	FileOutputStream* output = new FileOutputStream(fd);
	CHECK(google::protobuf::TextFormat::Print(proto, output));
	delete output;
	close(fd);
}

void ReadProtoFromBinaryFile(const char* filename, Message* proto) {
	int fd = open(filename, O_RDONLY);
	CHECK_NE(fd, -1)<< "File not found: " << filename;
	ZeroCopyInputStream* raw_input = new FileInputStream(fd);
	CodedInputStream* coded_input = new CodedInputStream(raw_input);
	coded_input->SetTotalBytesLimit(536870912, 268435456);

	CHECK(proto->ParseFromCodedStream(coded_input));

	delete coded_input;
	delete raw_input;
	close(fd);
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
	fstream output(filename, ios::out | ios::trunc | ios::binary);
	CHECK(proto.SerializeToOstream(&output));
}

int CreateDir(const char *sPathName, int beg) {
	char DirName[256];
	strcpy(DirName, sPathName);
	int i, len = strlen(DirName);
	if (DirName[len - 1] != '/')
		strcat(DirName, "/");

	len = strlen(DirName);

	for (i = beg; i < len; i++) {
		if (DirName[i] == '/') {
			DirName[i] = 0;
			if (access(DirName, 0) != 0) {
				CHECK(mkdir(DirName, 0755) == 0)
															<< "Failed to create folder "
															<< sPathName;
			}
//			if (access(DirName, NULL) != 0) {
//				if (mkdir(DirName, 0755) == -1) {
//					LOG(ERROR)<< "Failed to create folder " << sPathName;
//				}
//			}
			DirName[i] = '/';
		}
	}

	return 0;
}


}  // namespace caffe
