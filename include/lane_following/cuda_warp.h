#ifndef INCLUDE_LANE_FOLLOWING_CUDA_WARP_H__
#define INCLUDE_LANE_FOLLOWING_CUDA_WARP_H__

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda.inl.hpp>
#include <opencv2/cudawarping.hpp>

#include "lane_base.h"

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
		cv::cuda::warpPerspective(gpuImg, gpuOutImg, perspTf, gpuImg.size());
	}
private:
	cv::cuda::GpuMat gpuImg;
	cv::cuda::GpuMat gpuOutImg;
};

#endif /* INCLUDE_LANE_FOLLOWING_CUDA_WARP_H__ */
