#ifndef SRC_CUDAWARP_H_
#define SRC_CUDAWARP_H_

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda.inl.hpp>
#include <opencv2/cudawarping.hpp>

#include "LaneBase.h"

class CudaWarp: public WarpBase {
public:
	CudaWarp(int pipelineInstanceNum, bool bParallel, bool bGpuAccel,
			bool bVerbose) :
			WarpBase(pipelineInstanceNum, bParallel, bGpuAccel, bVerbose) {
	}
	virtual ~CudaWarp() {
		Deinit();
	}
	virtual void Init() override {
		gpuImg = cv::cuda::GpuMat(frameImg);
		WarpBase::Init();
	}
	virtual void Deinit() override {
		gpuImg.release();
		gpuOutImg.release();
		WarpBase::Deinit();
	}
	cv::cuda::GpuMat& getOutImg() {
		return gpuOutImg;
	}
	virtual void RunWarp() override {
		cv::cuda::warpPerspective(gpuImg, gpuOutImg, M, gpuImg.size());
	}
private:
	cv::cuda::GpuMat gpuImg;
	cv::cuda::GpuMat gpuOutImg;
};

#endif /* SRC_CUDAWARP_H_ */
