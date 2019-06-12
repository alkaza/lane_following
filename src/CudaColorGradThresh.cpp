#include "CudaColorGradThresh.h"

#include <opencv2/core/cuda.inl.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/imgproc.hpp>

#include "CudaWarp.h"

CudaColorGradThresh::CudaColorGradThresh(int pipelineInstanceNum,
		bool bParallel, bool bGpuAccel, bool bVerbose) :
		ColorGradThreshBase(pipelineInstanceNum, bParallel, bGpuAccel, bVerbose) {
}

CudaColorGradThresh::~CudaColorGradThresh() {
	Deinit();
}

void CudaColorGradThresh::Init() {
	ColorGradThreshBase::Init();
}

void CudaColorGradThresh::Deinit() {
	gpuImg.release();
	binarySrc.clear();
	binaryDst.clear();
	ColorGradThreshBase::Deinit();
}

void CudaColorGradThresh::setParams(LaneBase* obj) {
	Deinit();
	CudaWarp* warp = dynamic_cast<CudaWarp*>(obj);
	if (warp) {
		gpuImg = warp->getOutImg().clone();
		ColorGradThreshBase::setParams(obj);
	}
}

void CudaColorGradThresh::SplitChannel(SPLIT_MODE mode) {
	switch (mode) {
	case SPLIT_MODE_BGR: {
		// Separate BRG color channels
		cv::cuda::GpuMat bgrchannel[3];
		cv::cuda::split(gpuImg, bgrchannel);
		binarySrc.threshRed = bgrchannel[2];
		for (int i = 0; i < 3; i++) {
			bgrchannel[i].release();
		}
		break;
	}
	case SPLIT_MODE_HLS: {
		// Separate HLS color channels
		cv::cuda::GpuMat hlschannel[3];
		split(gpuHls, hlschannel);
		binarySrc.absSobelx = hlschannel[1];
		binarySrc.threshSat = hlschannel[2];
		for (int i = 0; i < 3; i++) {
			hlschannel[i].release();
		}
		gpuHls.release();
		break;
	}
	}
}

void CudaColorGradThresh::CvtBGR2HLS() {
	// Convert to HLS color space
	cv::cuda::cvtColor(gpuImg, gpuHls, cv::COLOR_BGR2HLS);
}

void CudaColorGradThresh::ThresholdBinary(THRESH_MODE mode) {
	switch (mode) {
	case THRESH_MODE_RED:
		Threshold(binarySrc.threshRed, thresh.red[0], thresh.red[1],
				binaryDst.threshRed);
		break;
	case THRESH_MODE_SAT:
		Threshold(binarySrc.threshSat, thresh.sat[0], thresh.sat[1],
				binaryDst.threshSat);
		break;
	case THRESH_MODE_ABS_SOBELX:
		Threshold(binarySrc.threshSobelx, thresh.sobelx[0], thresh.sobelx[1],
				binaryDst.threshSobelx);
		break;
	}
}

void CudaColorGradThresh::Threshold(const cv::cuda::GpuMat &src, int lowerb,
		int upperb, cv::cuda::GpuMat &dst) {
	cv::cuda::GpuMat res_high, res_low;
	cv::cuda::threshold(src, res_high, lowerb, 255, cv::THRESH_BINARY);
	cv::cuda::threshold(src, res_low, upperb, 255, cv::THRESH_BINARY_INV);
	cv::cuda::bitwise_and(res_high, res_low, dst);
	res_low.release();
	res_high.release();
}

void CudaColorGradThresh::Sobelx() {
	// Take the derivative in x
	cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createSobelFilter(
			binarySrc.absSobelx.type(),
			CV_16S, 1, 0);
	filter->apply(binarySrc.absSobelx, binaryDst.absSobelx);
}

void CudaColorGradThresh::AbsSobelx() {
	//Absolute x derivative to accentuate lines away from horizontal
	cv::cuda::abs(binaryDst.absSobelx, binaryDst.absSobelx);
	binaryDst.absSobelx.convertTo(binarySrc.threshSobelx, CV_8UC1);
}

void CudaColorGradThresh::CombBinaries() {
	// Combine three binary thresholds
	cv::cuda::bitwise_and(binaryDst.threshRed, binaryDst.threshSobelx,
			gpuOutImg);
	cv::cuda::bitwise_or(binaryDst.threshSat, gpuOutImg, gpuOutImg);
	gpuOutImg.download(outImg);
}
