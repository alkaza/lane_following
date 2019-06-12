#include "ThreadManager.h"

#include <opencv2/core/mat.inl.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <unistd.h>
#include <chrono>
#include <initializer_list>
#include <string>
#include <vector>

#include "ColorGradThresh.h"
#include "CompletedItem.h"
#include "CudaColorGradThresh.h"
#include "CudaWarp.h"
#include "Debug.h"
#include "LaneBase.h"
#include "Warp.h"

template class ThreadManager<CudaWarp, CudaColorGradThresh, FindLanes> ;
template class ThreadManager<Warp, ColorGradThresh, FindLanes> ;

template<typename WARP, typename COLOR_GRAD_THRESH, typename FIND_LANES>
ThreadManager<WARP, COLOR_GRAD_THRESH, FIND_LANES>::ThreadManager(tm_args *args) :
		args(args ? *args : tm_args()) {
	if (this->args.pipelineInstNum > MAX_PIPELINE_INST_NUM) {
		this->args.pipelineInstNum = MAX_PIPELINE_INST_NUM;
	}
	frameCnt = 0;
	pipelineFrameCnt = 0;
	processedFrameCnt = 0;

	bWarpTaskReady = false;
	bColorGradThreshTaskReady = false;
	bFindLanesTaskReady = false;
	bStartWarpTaskReady = false;

	frameDuration = 100000; //usec
	lastAngle = 0.5;
}

template<typename WARP, typename COLOR_GRAD_THRESH, typename FIND_LANES>
ThreadManager<WARP, COLOR_GRAD_THRESH, FIND_LANES>::~ThreadManager() {
}

template<typename WARP, typename COLOR_GRAD_THRESH, typename FIND_LANES>
void ThreadManager<WARP, COLOR_GRAD_THRESH, FIND_LANES>::Start() {
	PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_MANAGER, "++[%ld]ThreadManager::Start\n",
			GetThreadId());
	videoCap.open(args.videoFile);
	GetNextFrame();
	//imwrite("test.jpg", frameImg);

	std::string videoWrFile = "lane_detection";
	videoWrFile += std::to_string((int) args.speed);
	videoWrFile += ".avi";
	remove(videoWrFile.c_str());
	videoWr.open(videoWrFile, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 15,
			cv::Size(frameImg.cols, frameImg.rows));

	laneHistory.leftLine.found = true;
	laneHistory.rightLine.found = true;
	laneHistory.leftLine.xBase = 0;
	laneHistory.rightLine.xBase = frameImg.cols - 1;
	laneHistory.leftLine.fit = {0,0,0};
	laneHistory.rightLine.fit = {0,0,0};
	laneHistory.leftLine.angle = 0;
	laneHistory.rightLine.angle = 0;
	laneHistory.laneWidth = frameImg.cols - 1;

	StartWarp();
	PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_MANAGER, "--[%ld]ThreadManager::Start\n",
			GetThreadId());
}

template<typename WARP, typename COLOR_GRAD_THRESH, typename FIND_LANES>
void ThreadManager<WARP, COLOR_GRAD_THRESH, FIND_LANES>::Stop() {
	PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_MANAGER, "++[%ld]ThreadManager::Stop\n",
			GetThreadId());
	videoCap.release();
	videoWr.release();
	cv::destroyAllWindows();

	PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_MANAGER, "--[%ld]ThreadManager::Stop\n",
			GetThreadId());
}

template<typename WARP, typename COLOR_GRAD_THRESH, typename FIND_LANES>
void ThreadManager<WARP, COLOR_GRAD_THRESH, FIND_LANES>::GetNextFrame() {
	PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_MANAGER,
			"++[%ld]ThreadManager::GetNextFrame\n", GetThreadId());
	if (videoCap.isOpened()) {
		videoCap >> frameImg;
		if (frameImg.empty())
			PRINT_DEBUG_MSG(DEBUG_ZONE_ERROR,
					"[%ld]ThreadManager::GetNextFrame error: video frame is empty\n",
					GetThreadId());
	} else {
		PRINT_DEBUG_MSG(DEBUG_ZONE_ERROR,
				"[%ld]ThreadManager::GetNextFrame error: opening video stream or file\n",
				GetThreadId());
	}
	bStartWarpTaskReady = false;
	PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_MANAGER,
			"--[%ld]ThreadManager::GetNextFrame %dX%d\n", GetThreadId(),
			frameImg.rows, frameImg.cols);
}

template<typename WARP, typename COLOR_GRAD_THRESH, typename FIND_LANES>
inline void ThreadManager<WARP, COLOR_GRAD_THRESH, FIND_LANES>::ShowFrame(
		const cv::Mat &img) {
#if !DEBUG_ZONE_TEST
	if (args.bVerbose) {
		imshow("LaneDetection", img);
		cv::waitKey(1);
	}
#endif
	videoWr.write(img);
}

template<typename WARP, typename COLOR_GRAD_THRESH, typename FIND_LANES>
bool ThreadManager<WARP, COLOR_GRAD_THRESH, FIND_LANES>::StartTask(
		ThreadWorker* thread, CompletedItem &completedItem,
		std::shared_ptr<LaneBase> &obj) {
	PRINT_DEBUG_MSG((DEBUG_ZONE_THREAD_MANAGER || DEBUG_ZONE_PROCESS),
			"++[%ld->%ld]ThreadManager::StartTask: %s[%d], procStep = %s, frameIndex = %d\n",
			GetThreadId(), thread ? thread->GetThreadId() : -1,
			obj->getModuleName().c_str(), obj->getPipelineInstanceNum(),
			obj->getProcStepString(obj->getProcStep()), obj->getFrameIndex());

	if (thread) {
		completedItem.taskState = TASK_STATE_RUNNING;
		// Set thread to busy
		freeList.pop_front();
		busyList.push_back(thread);

		// Make message
		std::shared_ptr<ThreadMsg> msg = std::make_shared<ThreadMsg>();
		msg->msgObj = obj;
		switch (obj->msgObjType) {
		case MSG_OBJ_TYPE_WARP: {
			msg->taskMsg = ThreadWorker::TASK_MSG_RUN_WARP;
			break;
		}
		case MSG_OBJ_TYPE_COLOR_GRAD_THRESH: {
			msg->taskMsg = ThreadWorker::TASK_MSG_RUN_COLOR_GRAD_THRESH;
			break;
		}
		case MSG_OBJ_TYPE_FIND_LANES: {
			msg->taskMsg = ThreadWorker::TASK_MSG_RUN_FIND_LANES;
			break;
		}
		default:
			return false;
		}
		msg->procStep = completedItem.procStep;
		msg->threadIdFrom = GetThreadId();

		PRINT_DEBUG_MSG((DEBUG_ZONE_THREAD_MANAGER || DEBUG_ZONE_PROCESS),
				"--[%ld->%ld]ThreadManager::StartTask: %s[%d], procStep = %s, frameIndex = %d\n",
				GetThreadId(), thread ? thread->GetThreadId() : -1,
				obj->getModuleName().c_str(), obj->getPipelineInstanceNum(),
				obj->getProcStepString(obj->getProcStep()),
				obj->getFrameIndex());

		// Send message
		thread->AddMsg(msg);
		return true;
	}
	return false;
}

template<typename WARP, typename COLOR_GRAD_THRESH, typename FIND_LANES>
void ThreadManager<WARP, COLOR_GRAD_THRESH, FIND_LANES>::CompleteTask(
		std::shared_ptr<LaneBase> *obj) {
	PRINT_DEBUG_MSG((DEBUG_ZONE_THREAD_MANAGER || DEBUG_ZONE_PROCESS),
			"++[%ld]ThreadManager::CompleteTask: %s[%d], procStep = %d, frameIndex = %d\n",
			GetThreadId(), (*obj)->getModuleName().c_str(),
			(*obj)->getPipelineInstanceNum(), (*obj)->getProcStep(),
			(*obj)->getFrameIndex());

	bool* bTaskReady = nullptr;
	switch (obj[0]->msgObjType) {
	case MSG_OBJ_TYPE_WARP:
		bTaskReady = &bWarpTaskReady;
		break;
	case MSG_OBJ_TYPE_COLOR_GRAD_THRESH:
		bTaskReady = &bColorGradThreshTaskReady;
		break;
	case MSG_OBJ_TYPE_FIND_LANES:
		bTaskReady = &bFindLanesTaskReady;
		break;
	default:
		PRINT_DEBUG_MSG(DEBUG_ZONE_ERROR,
				"ThreadManager::CompleteTask error: unsupported msgObjType(%ud)\n",
				obj[0]->msgObjType)
		return;
	}
	*bTaskReady = false;

	for (int i = 0; i < args.pipelineInstNum; i++) {
		// If all step tasks are completed
		if (!obj[i]->completedItemList.empty()
				&& obj[i]->completedItemList.isCompleted()) {
			CompletedItemList old_completedItemList = obj[i]->completedItemList;
			int old_procStep = obj[i]->getProcStep();
			// Get next procStep
			obj[i]->NextStep();
			// If last procStep
			if (obj[i]->completedItemList.empty()) {
				bool startTask = false;
				PRINT_DEBUG_MSG(
						(DEBUG_ZONE_THREAD_MANAGER || DEBUG_ZONE_PROCESS),
						"[%ld]ThreadManager::CompleteTask: %s[%d] needs to start new task, next_procStep = %d, frameIndex = %d\n",
						GetThreadId(), obj[i]->getModuleName().c_str(),
						obj[i]->getPipelineInstanceNum(), obj[i]->getProcStep(),
						obj[i]->getFrameIndex());
				switch (obj[i]->msgObjType) {
				case MSG_OBJ_TYPE_WARP: {
					std::shared_ptr<WARP> temp_warp = std::dynamic_pointer_cast<
							WARP>(obj[i]);
					PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_MANAGER,
							"[%ld]ThreadManager::CompleteTask: WARP[%d] frame = %d DONE\n",
							GetThreadId(), temp_warp->getPipelineInstanceNum(),
							temp_warp->getFrameIndex());
					// If new module task didn't start
					if (temp_warp && !StartColorGradThresh(temp_warp)) {
						// Return to previous state
						temp_warp->completedItemList = old_completedItemList;
						temp_warp->setProcStep(old_procStep);
						// Marking that warp has uncompleted task
						bWarpTaskReady = true;
					} else {
						// New module task started
						startTask = true;
					}
					break;
				}
				case MSG_OBJ_TYPE_COLOR_GRAD_THRESH: {
					std::shared_ptr<COLOR_GRAD_THRESH> temp_colorGradThresh =
							std::dynamic_pointer_cast<COLOR_GRAD_THRESH>(
									obj[i]);
					PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_MANAGER,
							"[%ld]ThreadManager::CompleteTask: COLOR_GRAD_THRESH[%d] frame = %d DONE\n",
							GetThreadId(),
							temp_colorGradThresh->getPipelineInstanceNum(),
							temp_colorGradThresh->getFrameIndex());
					// If new task didn't start
					if (temp_colorGradThresh
							&& !StartFindLanes(temp_colorGradThresh)) {
						// Return to previous state
						temp_colorGradThresh->completedItemList =
								old_completedItemList;
						temp_colorGradThresh->setProcStep(old_procStep);
						// Marking that colorGradThresh has uncompleted task
						bColorGradThreshTaskReady = true;
					} else {
						// New task started
						startTask = true;
					}
					break;
				}
				case MSG_OBJ_TYPE_FIND_LANES: {
					std::shared_ptr<FIND_LANES> temp_findLanes =
							std::dynamic_pointer_cast<FIND_LANES>(obj[i]);
#if DEBUG_ZONE_TEST
					// Sorting frames
					if (temp_findLanes
							&& temp_findLanes->getFrameIndex()
									!= processedFrameCnt) {
						// Return to previous state
						temp_findLanes->completedItemList =
								old_completedItemList;
						temp_findLanes->setProcStep(old_procStep);
						// Marking that findLanes has uncompleted task
						bFindLanesTaskReady = true;
					} else {
						// All work on this frame is done
						startTask = true;
						PRINT_DEBUG_MSG(DEBUG_ZONE_FRAME,
								"[%ld]ThreadManager::CompleteTask: FIND_LANES[%d] frame = %d DONE\n",
								GetThreadId(),
								temp_findLanes->getPipelineInstanceNum(),
								temp_findLanes->getFrameIndex());
						ShowFrame(temp_findLanes->getOutImg());
						processedFrameCnt++;
						if (pipelineFrameCnt > 0) {
							pipelineFrameCnt--;
						}
						// Stop at last frame
						if (args.maxFrameCnt != -1
								&& temp_findLanes->getFrameIndex()
										== (args.maxFrameCnt - 1)) {
							EndThread();
						}
					}
#else
					if (temp_findLanes
							&& temp_findLanes->getFrameIndex()
									< processedFrameCnt) {
						PRINT_DEBUG_MSG(DEBUG_ZONE_FRAME,
								"[%ld]ThreadManager::CompleteTask: FIND_LANES[%d] frame = %d DONE, skipping\n",
								GetThreadId(),
								temp_findLanes->getPipelineInstanceNum(),
								temp_findLanes->getFrameIndex());
					} else {
						// All work on this frame is done
						usleep(args.delay); //increase frame processing duration
						frameDuration =
								std::chrono::duration_cast<
										std::chrono::microseconds>(
										std::chrono::high_resolution_clock::now()
												- temp_findLanes->getStartTime()).count();
						startTask = true;
						PRINT_DEBUG_MSG(DEBUG_ZONE_FRAME,
								"[%ld]ThreadManager::CompleteTask: FIND_LANES[%d] frame = %d DONE\n",
								GetThreadId(),
								temp_findLanes->getPipelineInstanceNum(),
								temp_findLanes->getFrameIndex());

						if (temp_findLanes->isDetected()) {
							lastAngle = temp_findLanes->getSteeringAngle();
							laneHistory = temp_findLanes->getLaneHistory();
						}

						ShowFrame(temp_findLanes->getOutImg());
						processedFrameCnt = temp_findLanes->getFrameIndex();

						// Stop at last frame
						if (args.maxFrameCnt != -1
								&& temp_findLanes->getFrameIndex()
										== (args.maxFrameCnt - 1)) {
							EndThread();
						}
					}
					if (pipelineFrameCnt > 0) {
						pipelineFrameCnt--;
					}
#endif
					break;
				}
				}
				PRINT_DEBUG_MSG(
						(DEBUG_ZONE_THREAD_MANAGER || DEBUG_ZONE_PROCESS),
						"[%ld]ThreadManager::CompleteTask: %s[%d] %s, next_procStep = %d, frameIndex = %d\n",
						GetThreadId(), obj[i]->getModuleName().c_str(),
						obj[i]->getPipelineInstanceNum(),
						(startTask ? "start task OK" : "cannot start task"),
						obj[i]->getProcStep(), obj[i]->getFrameIndex());
				// Always check to start new frame
				StartWarp();
			}
		}
		// Check if there are any unlaunched tasks
		if (!obj[i]->completedItemList.empty()) {
			int cnt = 0;
			for (auto &it : obj[i]->completedItemList) {
				if (it.taskState == TASK_STATE_INITIALIZED) {
					cnt++;
					// If we have free thread
					if (!freeList.empty()) {
						PRINT_DEBUG_MSG(
								(DEBUG_ZONE_THREAD_MANAGER || DEBUG_ZONE_PROCESS),
								"[%ld]ThreadManager::CompleteTask: %s[%d] has initialized task, procStep = %d, frameIndex = %d\n",
								GetThreadId(), obj[i]->getModuleName().c_str(),
								obj[i]->getPipelineInstanceNum(),
								obj[i]->getProcStep(), obj[i]->getFrameIndex());
						ThreadWorker* thread = freeList.front();
						// Start new task
						StartTask(thread, it, obj[i]);
					} else {
						*bTaskReady = true; // Mark that there are unlaunched tasks
						PRINT_DEBUG_MSG(
								(DEBUG_ZONE_THREAD_MANAGER || DEBUG_ZONE_PROCESS),
								"[%ld]ThreadManager::CompleteTask: %s[%d] no free threads, procStep = %d, frameIndex = %d, bWarpTaskReady = %d, bcolorGradThreshpTaskReady = %d, bFindLanesTaskReady = %d\n",
								GetThreadId(), obj[i]->getModuleName().c_str(),
								obj[i]->getPipelineInstanceNum(),
								obj[i]->getProcStep(), obj[i]->getFrameIndex(),
								bWarpTaskReady, bColorGradThreshTaskReady,
								bFindLanesTaskReady);
					}
				}

			}
			PRINT_DEBUG_MSG((DEBUG_ZONE_THREAD_MANAGER || DEBUG_ZONE_PROCESS),
					"[%ld]ThreadManager::CompleteTask: %s[%d] %s , procStep = %d, frameIndex = %d %s\n",
					GetThreadId(), obj[i]->getModuleName().c_str(),
					obj[i]->getPipelineInstanceNum(),
					(cnt ? "still has initialized tasks" : "has no initialized tasks"),
					obj[i]->getProcStep(), obj[i]->getFrameIndex(),
					obj[i]->completedItemList.dump().c_str());

		}
	}

	//if (bWarpTaskReady || bcolorGradThreshpTaskReady || bFindLanesTaskReady) {
	if (*bTaskReady) {
		// Make message
		std::shared_ptr<ThreadMsg> tmsg = std::make_shared<ThreadMsg>();
		tmsg->taskMsg = ThreadWorker::TASK_MSG_EXISTS_FIND_LANES;
		// Send unique message (to CompleteTask)
		AddUniqueMsg(tmsg);
		PRINT_DEBUG_MSG((DEBUG_ZONE_THREAD_MANAGER || DEBUG_ZONE_PROCESS),
				"[%ld]ThreadManager::CompleteTask: %s[%d] addUniqueMsg, procStep = %d, frameIndex = %d, bWarpTaskReady = %d, bColorGradThreshpTaskReady = %d, bFindLanesTaskReady = %d\n",
				GetThreadId(), (*obj)->getModuleName().c_str(),
				(*obj)->getPipelineInstanceNum(), (*obj)->getProcStep(),
				(*obj)->getFrameIndex(), bWarpTaskReady,
				bColorGradThreshTaskReady, bFindLanesTaskReady);

	}

	PRINT_DEBUG_MSG((DEBUG_ZONE_THREAD_MANAGER || DEBUG_ZONE_PROCESS),
			"--[%ld]ThreadManager::CompleteTask: %s[%d], procStep = %d, frameIndex = %d\n",
			GetThreadId(), (*obj)->getModuleName().c_str(),
			(*obj)->getPipelineInstanceNum(), (*obj)->getProcStep(),
			(*obj)->getFrameIndex());
}

template<typename WARP, typename COLOR_GRAD_THRESH, typename FIND_LANES>
bool ThreadManager<WARP, COLOR_GRAD_THRESH, FIND_LANES>::StartWarp() {
	PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_MANAGER,
			"++[%ld]ThreadManager::StartWarp\n", GetThreadId());
	bool ret = false;
	for (int i = 0;
			i < args.pipelineInstNum
					&& (frameCnt < args.maxFrameCnt || args.maxFrameCnt == -1)
					&& pipelineFrameCnt < args.pipelineInstNum; i++) {
		if (warp[i]->completedItemList.empty()) {
			GetNextFrame();
			auto startTime = std::chrono::high_resolution_clock::now();
			warp[i]->setStartTime(startTime);
			warp[i]->setFrameIndex(frameCnt);
			warp[i]->setFrameImg(frameImg);
			warp[i]->setParams(nullptr);
			for (auto &it : warp[i]->completedItemList) {
				if (it.taskState == TASK_STATE_INITIALIZED
						&& !freeList.empty()) {
					ThreadWorker* thread = freeList.front();
					std::shared_ptr<LaneBase> temp_warp(warp[i]);
					StartTask(thread, it, temp_warp);
				}
			}
			frameCnt++;
			pipelineFrameCnt++;
			ret = true;
		}
	}

	if (!ret) {
		if (frameCnt < args.maxFrameCnt || args.maxFrameCnt == -1) {
			bStartWarpTaskReady = true; // for debugging
		}
	}
	PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_MANAGER,
			"--[%ld]ThreadManager::StartWarp, ret = %d\n", GetThreadId(), ret);
	return ret;
}

template<typename WARP, typename COLOR_GRAD_THRESH, typename FIND_LANES>
bool ThreadManager<WARP, COLOR_GRAD_THRESH, FIND_LANES>::StartColorGradThresh(
		std::shared_ptr<WARP> &warp) {
	PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_MANAGER,
			"++[%ld]ThreadManager::StartColorGradThresh\n", GetThreadId());
	bool ret = false;
	for (int i = 0; i < args.pipelineInstNum && warp; i++) {
		if (colorGradThresh[i]->completedItemList.empty()) {
			colorGradThresh[i]->setStartTime(warp->getStartTime());
			colorGradThresh[i]->setFrameIndex(warp->getFrameIndex());
			colorGradThresh[i]->setFrameImg(warp->getFrameImg());
			colorGradThresh[i]->setInvPerspectiveTf(
					warp->getInvPerspectiveTf());
			colorGradThresh[i]->setParams(warp.get());
			for (auto &it : colorGradThresh[i]->completedItemList) {
				if (it.taskState == TASK_STATE_INITIALIZED
						&& !freeList.empty()) {
					ThreadWorker* thread = freeList.front();
					std::shared_ptr<LaneBase> temp_colorGradThresh(
							colorGradThresh[i]);
					StartTask(thread, it, temp_colorGradThresh);
				}
			}
			ret = true;
			break;
		}
	}
	PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_MANAGER,
			"--[%ld]ThreadManager::StartColorGradThresh, ret = %d\n",
			GetThreadId(), ret);
	return ret;
}

template<typename WARP, typename COLOR_GRAD_THRESH, typename FIND_LANES>
bool ThreadManager<WARP, COLOR_GRAD_THRESH, FIND_LANES>::StartFindLanes(
		std::shared_ptr<COLOR_GRAD_THRESH> &colorGradThresh) {
	PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_MANAGER,
			"++[%ld]ThreadManager::StartFindLanes\n", GetThreadId());
	bool ret = false;
	for (int i = 0; i < args.pipelineInstNum && colorGradThresh; i++) {
		if (findLanes[i]->completedItemList.empty()) {
			findLanes[i]->setStartTime(colorGradThresh->getStartTime());
			findLanes[i]->setFrameDuration(frameDuration);
			findLanes[i]->setSpeed(args.speed);
			findLanes[i]->setLaneHistory(laneHistory);
			findLanes[i]->setFrameIndex(colorGradThresh->getFrameIndex());
			findLanes[i]->setFrameImg(colorGradThresh->getFrameImg());
			findLanes[i]->setInvPerspectiveTf(
					colorGradThresh->getInvPerspectiveTf());
			findLanes[i]->setParams(colorGradThresh.get());
			for (auto &it : findLanes[i]->completedItemList) {
				if (it.taskState == TASK_STATE_INITIALIZED
						&& !freeList.empty()) {
					ThreadWorker* thread = freeList.front();
					std::shared_ptr<LaneBase> temp_findLanes(findLanes[i]);
					StartTask(thread, it, temp_findLanes);
				}
			}
			ret = true;
			break;
		}
	}
	PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_MANAGER,
			"--[%ld]ThreadManager::StartFindLanes, ret = %d\n", GetThreadId(),
			ret);
	return ret;
}

template<typename WARP, typename COLOR_GRAD_THRESH, typename FIND_LANES>
void ThreadManager<WARP, COLOR_GRAD_THRESH, FIND_LANES>::ProcessMsg(
		std::shared_ptr<ThreadMsg> &msg) {
	switch (msg->taskMsg) {
	case THREAD_MSG_START: {
		PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_MANAGER,
				"++[%ld]ThreadManager::ProcessMsg: THREAD_MSG_START received\n",
				GetThreadId());
		Start();
		PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_MANAGER,
				"--[%ld]ThreadManager::ProcessMsg: THREAD_MSG_START received\n",
				GetThreadId());
		break;
	}
	case ThreadWorker::TASK_MSG_COMPLETE_WARP: {
		PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_MANAGER,
				"++[%ld]ThreadManager::ProcessMsg: TASK_MSG_COMPLETE_WARP received\n",
				GetThreadId());
		// Free completed thread
		ThreadWorker* thread = getBusyThread(msg->threadIdFrom);
		if (thread) {
			freeList.push_back(thread);
		} else {
			PRINT_DEBUG_MSG(DEBUG_ZONE_ERROR,
					"[%ld]ThreadManager::ProcessMsg error: TASK_MSG_COMPLETE_WARP received, thread[%ld] not found in busyList\n",
					GetThreadId(), msg->threadIdFrom);
		}
		// Add CompletedItem
		WARP* temp_warp = dynamic_cast<WARP*>(msg->msgObj.get());
		if (temp_warp) {
			temp_warp->completedItemList.addItem(msg->procStep,
					TASK_STATE_COMPLETED);
			PRINT_DEBUG_MSG(DEBUG_ZONE_PROCESS,
					"[%ld]ThreadManager::ProcessMsg: warp[%d].completedItemList: %s\n",
					GetThreadId(), temp_warp->getPipelineInstanceNum(),
					temp_warp->completedItemList.dump().c_str());
		} else {
			PRINT_DEBUG_MSG(DEBUG_ZONE_ERROR,
					"[%ld]ThreadManager::ProcessMsg: TASK_MSG_COMPLETE_WARP received, temp_warp = NULL\n",
					GetThreadId());

		}
		// Call CompleteTask()
		CompleteTask(reinterpret_cast<std::shared_ptr<LaneBase>*>(warp));
		if (bColorGradThreshTaskReady) {
			CompleteTask(
					reinterpret_cast<std::shared_ptr<LaneBase>*>(colorGradThresh));
		}
		if (bFindLanesTaskReady) {
			CompleteTask(
					reinterpret_cast<std::shared_ptr<LaneBase>*>(findLanes));
		}
		PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_MANAGER,
				"--[%ld]ThreadManager::ProcessMsg: TASK_MSG_COMPLETE_WARP received\n",
				GetThreadId());
		break;
	}
	case ThreadWorker::TASK_MSG_COMPLETE_COLOR_GRAD_THRESH: {
		PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_MANAGER,
				"++[%ld]ThreadManager::ProcessMsg: TASK_MSG_COMPLETE_COLOR_GRAD_THRESH received\n",
				GetThreadId());
		// Free completed thread
		ThreadWorker* thread = getBusyThread(msg->threadIdFrom);
		if (thread) {
			freeList.push_back(thread);
		} else {
			PRINT_DEBUG_MSG(DEBUG_ZONE_ERROR,
					"[%ld]ThreadManager::ProcessMsg error: TASK_MSG_COMPLETE_COLOR_GRAD_THRESH received, thread[%ld] not found in busyList\n",
					GetThreadId(), msg->threadIdFrom);
		}
		// Add completed item
		COLOR_GRAD_THRESH* temp_colorGradThresh =
				dynamic_cast<COLOR_GRAD_THRESH*>(msg->msgObj.get());
		if (temp_colorGradThresh) {
			temp_colorGradThresh->completedItemList.addItem(msg->procStep,
					TASK_STATE_COMPLETED);
			PRINT_DEBUG_MSG(DEBUG_ZONE_PROCESS,
					"[%ld]ThreadManager::ProcessMsg: temp_colorGradThresh[%d].completedItemList: %s\n",
					GetThreadId(),
					temp_colorGradThresh->getPipelineInstanceNum(),
					temp_colorGradThresh->completedItemList.dump().c_str());
		} else {
			PRINT_DEBUG_MSG(DEBUG_ZONE_ERROR,
					"[%ld]ThreadManager::ProcessMsg: TASK_MSG_COMPLETE_COLOR_GRAD_THRESH received, temp_colorGradThresh = NULL\n",
					GetThreadId());
		}
		// Call CompleteTask()
		CompleteTask(
				reinterpret_cast<std::shared_ptr<LaneBase>*>(colorGradThresh));
		if (bFindLanesTaskReady) {
			CompleteTask(
					reinterpret_cast<std::shared_ptr<LaneBase>*>(findLanes));
		}
		if (bWarpTaskReady) {
			CompleteTask(reinterpret_cast<std::shared_ptr<LaneBase>*>(warp));
		}
		PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_MANAGER,
				"--[%ld]ThreadManager::ProcessMsg: TASK_MSG_COMPLETE_COLOR_GRAD_THRESH received\n",
				GetThreadId());
		break;
	}
	case ThreadWorker::TASK_MSG_COMPLETE_FIND_LANES: {
		PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_MANAGER,
				"++[%ld]ThreadManager::ProcessMsg: TASK_MSG_COMPLETE_FIND_LANES received\n",
				GetThreadId());
		ThreadWorker* thread = getBusyThread(msg->threadIdFrom);
		// Free completed thread
		if (thread) {
			freeList.push_back(thread);
		} else {
			PRINT_DEBUG_MSG(DEBUG_ZONE_ERROR,
					"[%ld]ThreadManager::ProcessMsg error: TASK_MSG_COMPLETE_FIND_LANES received, thread[%ld] not found in busyList\n",
					GetThreadId(), msg->threadIdFrom);
		}
		// Add completed item
		FIND_LANES* temp_findLanes =
				dynamic_cast<FIND_LANES*>(msg->msgObj.get());
		if (temp_findLanes) {
			temp_findLanes->completedItemList.addItem(msg->procStep,
					TASK_STATE_COMPLETED);
			PRINT_DEBUG_MSG(DEBUG_ZONE_PROCESS,
					"[%ld]ThreadManager::ProcessMsg: findLanes[%d].completedItemList: %s\n",
					GetThreadId(), temp_findLanes->getPipelineInstanceNum(),
					temp_findLanes->completedItemList.dump().c_str());
		} else {
			PRINT_DEBUG_MSG(DEBUG_ZONE_ERROR,
					"[%ld]ThreadManager::ProcessMsg: TASK_MSG_COMPLETE_FIND_LANES received, temp_findLanes = NULL\n",
					GetThreadId());

		}
		// Call CompleteTask()
		CompleteTask(reinterpret_cast<std::shared_ptr<LaneBase>*>(findLanes));
		if (bWarpTaskReady) {
			CompleteTask(reinterpret_cast<std::shared_ptr<LaneBase>*>(warp));
		}
		if (bColorGradThreshTaskReady) {
			CompleteTask(
					reinterpret_cast<std::shared_ptr<LaneBase>*>(colorGradThresh));
		}
		PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_MANAGER,
				"--[%ld]ThreadManager::ProcessMsg: TASK_MSG_COMPLETE_FIND_LANES received\n",
				GetThreadId());
		break;
	}
	case ThreadWorker::TASK_MSG_EXISTS_FIND_LANES: {
		if (bWarpTaskReady)
			CompleteTask(reinterpret_cast<std::shared_ptr<LaneBase>*>(warp));
		if (bColorGradThreshTaskReady)
			CompleteTask(
					reinterpret_cast<std::shared_ptr<LaneBase>*>(colorGradThresh));
		if (bFindLanesTaskReady)
			CompleteTask(
					reinterpret_cast<std::shared_ptr<LaneBase>*>(findLanes));
		break;
	}
	}
}

template<typename WARP, typename COLOR_GRAD_THRESH, typename FIND_LANES>
bool ThreadManager<WARP, COLOR_GRAD_THRESH, FIND_LANES>::PreWorkInit() {
	// Create module instances
	for (int i = 0; i < args.pipelineInstNum; i++) {
		warp[i] = std::make_shared<WARP>(i, args.bParallel, args.bGpuAccel,
				args.bVerbose);
		colorGradThresh[i] = std::make_shared<COLOR_GRAD_THRESH>(i,
				args.bParallel, args.bGpuAccel, args.bVerbose);
		findLanes[i] = std::make_shared<FIND_LANES>(i, args.bParallel,
				args.bGpuAccel, args.bVerbose);

		PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_MANAGER,
				"[%ld]ThreadManager::PreWorkInit, warp[%d]=%p\n", GetThreadId(),
				i, warp[i].get());
		PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_MANAGER,
				"[%ld]ThreadManager::PreWorkInit, colorGradThresh[%d]=%p\n",
				GetThreadId(), i, colorGradThresh[i].get());
		PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_MANAGER,
				"[%ld]ThreadManager::PreWorkInit, findLanes[%d]=%p\n",
				GetThreadId(), i, findLanes[i].get());
	}

	// Create thread pool
	for (int i = 0; i < args.threadPoolSize; i++) {
		ThreadWorker* thread = new ThreadWorker();
		if (thread->ThreadCreate()) {
			PRINT_DEBUG_MSG(DEBUG_ZONE_THREAD_MANAGER,
					"[%ld]ThreadManager::PreWorkInit, thread[%ld] created\n",
					GetThreadId(), thread->GetThreadId());
		} else {
			PRINT_DEBUG_MSG(DEBUG_ZONE_ERROR,
					"[%ld]ThreadManager::PreWorkInit error: failed creating thread[%ld]\n",
					GetThreadId(), thread->GetThreadId());
		}
		freeList.push_back(thread);
	}
	return true;
}

template<typename WARP, typename COLOR_GRAD_THRESH, typename FIND_LANES>
bool ThreadManager<WARP, COLOR_GRAD_THRESH, FIND_LANES>::PostWorkDeinit() {
	// Wait for threads to complete
	int cnt = 100;
	while (!busyList.empty() && --cnt > 0) {
		usleep(10000);
	}
	if (!busyList.empty()) {
		PRINT_DEBUG_MSG(DEBUG_ZONE_ERROR,
				"[%ld]ThreadManager::PostWorkDeinit error: %ld threads still running\n",
				GetThreadId(), busyList.size());
	}
	// Wrap up
	Stop();
	// Terminate threads
	for (auto it = freeList.begin(); it != freeList.end(); it++) {
		(*it)->EndThread();
		delete *it;
	}
	freeList.clear();
	for (auto it = busyList.begin(); it != busyList.end(); it++) {
		(*it)->EndThread();
		delete *it;
	}
	busyList.clear();
	return true;
}

template<typename WARP, typename COLOR_GRAD_THRESH, typename FIND_LANES>
inline void ThreadManager<WARP, COLOR_GRAD_THRESH, FIND_LANES>::PrintAvgDurations() {
	warp[0]->MakeDurations();
	for (int i = 1; i < args.pipelineInstNum; i++) {
		warp[i]->MakeDurations();
		warp[0]->AppendDurations(warp[i]->getDurations());
	}
	warp[0]->PrintAvgDurations("Warp");

	colorGradThresh[0]->MakeDurations();
	for (int i = 1; i < args.pipelineInstNum; i++) {
		colorGradThresh[i]->MakeDurations();
		colorGradThresh[0]->AppendDurations(colorGradThresh[i]->getDurations());
	}
	colorGradThresh[0]->PrintAvgDurations("colorGradThresh");

	findLanes[0]->MakeDurations();
	for (int i = 1; i < args.pipelineInstNum; i++) {
		findLanes[i]->MakeDurations();
		findLanes[0]->AppendDurations(findLanes[i]->getDurations());
	}
	findLanes[0]->PrintAvgDurations("FindLanes");
}

