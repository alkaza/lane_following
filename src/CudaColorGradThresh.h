#ifndef SRC_CUDACOLORGRADTHRESH_H_
#define SRC_CUDACOLORGRADTHRESH_H_

#include <opencv2/core/cuda.hpp>

#include "LaneBase.h"

class CudaColorGradThresh: public ColorGradThreshBase {
public:
	CudaColorGradThresh(int pipelineInstanceNum, bool bParallel, bool bGpuAccel,
			bool bVerbose);
	virtual ~CudaColorGradThresh();
	virtual void Init() override;
	virtual void Deinit() override;
	virtual void setParams(LaneBase* obj) override;
protected:
	virtual void SplitChannel(SPLIT_MODE mode) override;
	virtual void CvtBGR2HLS() override;
	virtual void ThresholdBinary(THRESH_MODE mode) override;
	virtual void Sobelx() override;
	virtual void AbsSobelx() override;
	virtual void CombBinaries() override;
private:
	cv::cuda::GpuMat gpuImg;
	cv::cuda::GpuMat gpuOutImg;
	cv::cuda::GpuMat gpuHls;
	struct Binaries {
		cv::cuda::GpuMat threshRed;
		cv::cuda::GpuMat threshSat;
		cv::cuda::GpuMat absSobelx;
		cv::cuda::GpuMat threshSobelx;
		void clear() {
			threshRed.release();
			threshSat.release();
			absSobelx.release();
			threshSobelx.release();
		}
	} binarySrc, binaryDst;
	void Threshold(const cv::cuda::GpuMat &src, int lowerb, int upperb,
			cv::cuda::GpuMat &dst);
};

#endif /* SRC_CUDACOLORGRADTHRESH_H_ */
