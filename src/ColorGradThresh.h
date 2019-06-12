#ifndef SRC_COLORGRADTHRESH_H_
#define SRC_COLORGRADTHRESH_H_

#include <opencv2/core/mat.hpp>
#include <opencv2/core/mat.inl.hpp>

#include "LaneBase.h"

class ColorGradThresh: public ColorGradThreshBase {
public:
	ColorGradThresh(int pipelineInstanceNum, bool bParallel, bool bGpuAccel,
			bool bVerbose);
	virtual ~ColorGradThresh();
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
	cv::Mat img;
	cv::Mat hls;
	struct Binaries {
		cv::Mat threshRed;
		cv::Mat threshSat;
		cv::Mat absSobelx;
		cv::Mat threshSobelx;
		void clear() {
			threshRed.release();
			threshSat.release();
			absSobelx.release();
			threshSobelx.release();
		}
	} binarySrc, binaryDst;
	void Threshold(const cv::Mat &src, int lowerb, int upperb, cv::Mat &dst);
};

#endif /* SRC_COLORGRADTHRESH_H_ */
