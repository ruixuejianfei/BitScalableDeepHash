// Copyright 2013 Yangqing Jia

#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_

#include <boost/shared_ptr.hpp>
#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>
// cuda driver types
#include <driver_types.h>
#include <glog/logging.h>
#include <mkl_vsl.h>
#include <map>
#include <string>

// various checks for different function calls.
#define CUDA_CHECK(condition) CHECK_EQ((condition), cudaSuccess)
#define CUBLAS_CHECK(condition) CHECK_EQ((condition), CUBLAS_STATUS_SUCCESS)
#define CURAND_CHECK(condition) CHECK_EQ((condition), CURAND_STATUS_SUCCESS)
#define VSL_CHECK(condition) CHECK_EQ((condition), VSL_STATUS_OK)

// After a kernel is executed, this will check the error and if there is one,
// exit loudly.
#define CUDA_POST_KERNEL_CHECK \
  if (cudaSuccess != cudaPeekAtLastError()) \
    LOG(FATAL) << "Cuda kernel failed. Error: " \
        << cudaGetErrorString(cudaPeekAtLastError())

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)

// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname) \
  template class classname<float>; \
  template class classname<double>

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

//#include <iostream>
//#define ZHU_FILE (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
//#define ZHU_PRINT  std::cout << "\n" <<  __TIME__ << " [" << ZHU_FILE << ":" << __LINE__ << "] "

namespace caffe {

// We will use the boost shared_ptr instead of the new C++11 one mainly
// because cuda does not work (at least now) well with C++11 features.
using boost::shared_ptr;

// We will use 1024 threads per block, which requires cuda sm_2x or above.
#if __CUDA_ARCH__ >= 200
const int CAFFE_CUDA_NUM_THREADS = 1024;
#else
const int CAFFE_CUDA_NUM_THREADS = 512;
#endif

inline int CAFFE_GET_BLOCKS(const int N) {
	return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

// A singleton class to hold common caffe stuff, such as the handler that
// caffe is going to use for cublas, curand, etc.
class Caffe {
public:
	~Caffe();
	inline static Caffe& Get() {
		if (!singleton_.get()) {
			singleton_.reset(new Caffe());
		}
		return *singleton_;
	}
	enum Brew {
		CPU, GPU
	};
	enum Phase {
		TRAIN, TEST
	};

	// The getters for the variables.
	// Returns the cublas handle.
	inline static cublasHandle_t cublas_handle() {
		return Get().cublas_handle_;
	}
	// Returns the curand generator.
	inline static curandGenerator_t curand_generator() {
		return Get().curand_generator_;
	}
	// Returns the MKL random stream.
	inline static VSLStreamStatePtr vsl_stream() {
		return Get().vsl_stream_;
	}
	// Returns the mode: running on CPU or GPU.
	inline static Brew mode() {
		return Get().mode_;
	}
	// Returns the phase: TRAIN or TEST.
	inline static Phase phase() {
		return Get().phase_;
	}
	// The setters for the variables
	// Sets the mode. It is recommended that you don't change the mode halfway
	// into the program since that may cause allocation of pinned memory being
	// freed in a non-pinned way, which may cause problems - I haven't verified
	// it personally but better to note it here in the header file.
	inline static void set_mode(Brew mode) {
		Get().mode_ = mode;
	}
	// Sets the phase.
	inline static void set_phase(Phase phase) {
		Get().phase_ = phase;
	}

	// zhu
	inline static bool datalayer_remain() {
		return Get().datalayer_remain_;
	}
	inline static void set_datalayer_remain(bool datalayer_remain) {
		Get().datalayer_remain_ = datalayer_remain;
	}

	inline static std::map<std::string, int>& mutable_name2id() {
		return Get().name2id_;
	}

	inline static std::vector<std::vector<std::string> >& mutable_triplets() {
		return Get().triplets_;
	}

	inline static std::vector<std::vector<int> >& mutable_triplets_id() {
		return Get().triplets_id_;
	}

	inline static int& mutable_pos_triplets() {
		return Get().pos_triplets_;
	}

	inline static std::map<std::string, int>& mutable_prefetch_name2id() {
		return Get().prefetch_name2id_;
	}

	inline static std::vector<int>& mutable_imgclass() {
		return Get().imgclass_;
	}

	inline static std::vector<int>& mutable_prefetch_imgclass() {
		return Get().prefetch_imgclass_;
	}

	inline static std::vector<std::vector<std::string> >& mutable_prefetch_triplets() {
		return Get().prefetch_triplets_;
	}

	inline static std::vector<std::vector<int> >& mutable_prefetch_triplets_id() {
		return Get().prefetch_triplets_id_;
	}


	// Sets the random seed of both MKL and curand
	static void set_random_seed(const unsigned int seed);
	// Sets the device. Since we have cublas and curand stuff, set device also
	// requires us to reset those values.
	static void SetDevice(const int device_id);
	// Prints the current GPU status.
	static void DeviceQuery();

protected:
	cublasHandle_t cublas_handle_;
	curandGenerator_t curand_generator_;
	VSLStreamStatePtr vsl_stream_;
	Brew mode_;
	Phase phase_;
	// zhu
	// 在做扰动的时候，并不想预读取数据，想读取当前batch多次，那么可以把这个设为true.
	bool datalayer_remain_;
	// 产生triplet的时候，可能出现重复的图片，用这个来记录，避免重复图片重复计算，减少开销
	// name是图片的名字，id则是图片在blob对应的位置(id*img_size)
	std::map<std::string, int> name2id_;
	std::map<std::string, int> prefetch_name2id_;
	// 所有的triplet
	std::vector<std::vector<std::string> > triplets_;
	std::vector<std::vector<std::string> > prefetch_triplets_;
	// 所有的triplet对应的id，见name2id_
	std::vector<std::vector<int> > triplets_id_;
	std::vector<std::vector<int> > prefetch_triplets_id_;
	std::vector<int> imgclass_;
	std::vector<int> prefetch_imgclass_;

	int pos_triplets_;
	static shared_ptr<Caffe> singleton_;

private:
	// The private constructor to avoid duplicate instantiation.
	Caffe();

DISABLE_COPY_AND_ASSIGN(Caffe);
};

}  // namespace caffe

#endif  // CAFFE_COMMON_HPP_
