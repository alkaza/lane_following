#ifndef SRC_WARP_H_
#define SRC_WARP_H_

#include <opencv2/core/mat.hpp>
#include <opencv2/core/mat.inl.hpp>
#include <opencv2/imgproc.hpp>

#include "LaneBase.h"

class Warp: public WarpBase {
public:
	Warp(int pipelineInstanceNum, bool bParallel, bool bGpuAccel, bool bVerbose) :
			WarpBase(pipelineInstanceNum, bParallel, bGpuAccel, bVerbose) {
	}
	virtual ~Warp() {
		Deinit();
	}
	virtual void Init() override {
		WarpBase::Init();
	}
	virtual void Deinit() override {
		outImg.release();
		WarpBase::Deinit();
	}
	cv::Mat& getOutImg() {
		return outImg;
	}
	virtual void RunWarp() override {
		warpPerspective(frameImg, outImg, M, frameImg.size());
	}
private:
	cv::Mat outImg;
};

#endif /* SRC_WARP_H_ */
