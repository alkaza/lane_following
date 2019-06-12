#ifndef SRC_THREADMANAGER_H_
#define SRC_THREADMANAGER_H_

#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <cstdio>
#include <list>
#include <memory>

#include "FindLanes.h"
#include "ThreadBase.h"
#include "ThreadWorker.h"

struct CompletedItem;
class LaneBase;

struct tm_args {
	cv::String videoFile;
	int threadPoolSize;
	int pipelineInstNum;
	int maxFrameCnt;
	double speed;
	long int delay;
	bool bParallel;
	bool bGpuAccel;
	bool bVerbose;
	tm_args() {
		videoFile = "project_video.mp4";
		threadPoolSize = 8;
		pipelineInstNum = 3;
		maxFrameCnt = 100;
		speed = 1000;
		delay = 0;
		bParallel = false;
		bGpuAccel = false;
		bVerbose = false;
	}
};

template<typename WARP, typename COLOR_GRAD_THRESH, typename FIND_LANES>
class ThreadManager: public ThreadBase {
public:
	enum {
		MAX_PIPELINE_INST_NUM = 16
	};
	ThreadManager(tm_args *args);
	virtual ~ThreadManager();
	void Start();
	void Stop();
	void GetNextFrame();
	void ShowFrame(const cv::Mat &img);
	bool StartTask(ThreadWorker* thread, CompletedItem &complete_item,
			std::shared_ptr<LaneBase> &obj);
	void CompleteTask(std::shared_ptr<LaneBase> *obj);
	bool StartWarp();
	bool StartColorGradThresh(std::shared_ptr<WARP> &warp);
	bool StartFindLanes(std::shared_ptr<COLOR_GRAD_THRESH> &colorGradThresh);
	virtual void ProcessMsg(std::shared_ptr<ThreadMsg> &msg) override;
	virtual bool PreWorkInit() override;
	virtual bool PostWorkDeinit() override;
	void PrintAvgDurations();
	ThreadWorker* getBusyThread(ThreadId threadId) {
		for (auto it = busyList.begin(); it != busyList.end(); it++) {
			if ((*it)->GetThreadId() == threadId) {
				ThreadWorker* r = *it;
				busyList.erase(it);
				return r;
			}
		}
		return NULL;
	}

private:
	static ThreadManager<WARP, COLOR_GRAD_THRESH, FIND_LANES>* threadManager;
	tm_args args;

	cv::VideoCapture videoCap;
	cv::Mat frameImg;

	std::list<ThreadWorker*> freeList; // list of free threads
	std::list<ThreadWorker*> busyList; // list of busy threads

	std::shared_ptr<WARP> warp[MAX_PIPELINE_INST_NUM];
	std::shared_ptr<COLOR_GRAD_THRESH> colorGradThresh[MAX_PIPELINE_INST_NUM];
	std::shared_ptr<FIND_LANES> findLanes[MAX_PIPELINE_INST_NUM];

	int frameCnt;
	int pipelineFrameCnt;
	int processedFrameCnt;

	bool bWarpTaskReady;
	bool bColorGradThreshTaskReady;
	bool bFindLanesTaskReady;
	bool bStartWarpTaskReady;

	long int frameDuration;
	double lastAngle;
	FindLanes::LaneHistory laneHistory;

	cv::VideoWriter videoWr;
};

#endif /* SRC_THREADMANAGER_H_ */
