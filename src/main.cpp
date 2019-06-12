/*
 * main.cpp
 *
 *  Created on: Mar 1, 2019
 *      Author: Alena Kazakova
 */

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda.inl.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/mat.inl.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stddef.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>

#include "ColorGradThresh.h"
#include "CudaColorGradThresh.h"
#include "CudaWarp.h"
#include "Debug.h"
#include "FindLanes.h"
#include "LaneBase.h"
#include "ThreadBase.h"
#include "ThreadManager.h"
#include "Warp.h"

void CudaInit() {
	cv::Mat img;
	img = cv::Mat::zeros(720, 1280, CV_8UC3);
	cv::cuda::GpuMat gpu_img = cv::cuda::GpuMat(img);
	CudaWarp* warp = new CudaWarp(0, false, false, false);
	warp->setFrameImg(img);
	warp->RunWarp();
	delete warp;
	gpu_img.release();
}

void RunWarp() {
	cv::Mat img, out_img;
	img = cv::imread("frame_img.jpg", cv::IMREAD_COLOR);
	Warp* warp = new Warp(0, false, false, false);
	warp->setFrameImg(img);
	warp->RunWarp();
	out_img = warp->getOutImg();
	cv::imwrite("warped_img.jpg", out_img);
	delete warp;
}

void Run(tm_args *args) {
	ThreadManager<CudaWarp, CudaColorGradThresh, FindLanes>* threadManagerCuda =
	NULL;
	ThreadManager<Warp, ColorGradThresh, FindLanes>* threadManager = NULL;

	if (args->bGpuAccel) {
#if !DEBUG_ZONE_TEST
		CudaInit();
#endif
		threadManagerCuda = new ThreadManager<CudaWarp, CudaColorGradThresh,
				FindLanes>(args);
	} else {
		threadManager = new ThreadManager<Warp, ColorGradThresh, FindLanes>(
				args);
	}

	std::shared_ptr<ThreadMsg> msg = std::make_shared<ThreadMsg>();
	msg->taskMsg = ThreadBase::THREAD_MSG_START;

	if (threadManagerCuda && !threadManagerCuda->ThreadCreate()) {
		PRINT_DEBUG_MSG(DEBUG_ZONE_ERROR, "Error creating thread manager\n")
	}
	if (threadManager && !threadManager->ThreadCreate()) {
		PRINT_DEBUG_MSG(DEBUG_ZONE_ERROR, "Error creating thread manager\n")
	}

	auto start_time = std::chrono::high_resolution_clock::now();

	if (threadManagerCuda) {
		threadManagerCuda->AddMsg(msg);
		threadManagerCuda->WaitForThread();
	}
	if (threadManager) {
		threadManager->AddMsg(msg);
		threadManager->WaitForThread();
	}

	auto end_time = std::chrono::high_resolution_clock::now();
	auto exec_duration = std::chrono::duration_cast<std::chrono::microseconds>(
			end_time - start_time);
#if DEBUG_ZONE_TEST
	std::cout << std::left << std::setw(8)
			<< exec_duration.count() / args->maxFrameCnt;
#else
	std::cout << "<" << (args->bParallel ? "Parallel" : "")
			<< (args->bGpuAccel ? "GpuAccel" : "")
			<< (args->pipelineInstNum > 1 ? "Pipeline" : "Sequential") << ">"
			<< std::endl;
	std::cout << std::left << std::setw(20) << "Threads" << args->threadPoolSize
			<< std::endl;
	std::cout << std::left << std::setw(20) << "Pipeline"
			<< args->pipelineInstNum << std::endl;
	std::cout << std::left << std::setw(20) << "Frames" << args->maxFrameCnt
			<< std::endl;
	std::cout << std::left << std::setw(20) << "Execution time" << std::setw(12)
			<< exec_duration.count() << " usec" << std::endl;
	std::cout << std::left << std::setw(20) << "Reaction time" << std::setw(12)
			<< exec_duration.count() / args->maxFrameCnt << " usec" << std::endl
			<< std::endl;
	if (threadManagerCuda) {
		threadManagerCuda->PrintAvgDurations();
	}
	if (threadManager) {
		threadManager->PrintAvgDurations();
	}
#endif
	delete threadManagerCuda;
	delete threadManager;
	ThreadBase::ResetThreadRegister();

}
void TestHelper(tm_args *args) {
	std::cout << "<" << (args->bParallel ? "Parallel" : "")
			<< (args->bGpuAccel ? "GpuAccel" : "")
			<< (args->pipelineInstNum > 1 ? "Pipeline" : "Sequential") << ">"
			<< std::endl;
	for (int i = 0; i < 9; i++) {
		std::cout << std::left << std::setw(8) << i;
	}
	std::cout << std::endl;

	for (int i = 1; i < 9; i++) {
		std::cout << std::setw(8) << i;
		args->threadPoolSize = i;
		for (int j = 1; j < 9; j++) {
			args->pipelineInstNum = j;
			Run(args);
		}
		std::cout << std::endl;
	}
}

void Test() {
	tm_args args;
	args.videoFile = "project_video.mp4";
	args.maxFrameCnt = 100;
	args.bVerbose = true;

	args.bGpuAccel = false;
	args.bParallel = false;
	TestHelper(&args);
	args.bParallel = true;
	TestHelper(&args);

	args.bGpuAccel = true;
	args.bParallel = false;
	CudaInit();
	TestHelper(&args);
	args.bParallel = true;
	TestHelper(&args);
}

int main(int argc, char** argv) {
#if DEBUG_ZONE_TEST
	Test();
#else
	tm_args args;
	args.videoFile = "raw_video.avi";
	args.maxFrameCnt = 100;
	args.pipelineInstNum = 1;
	args.bVerbose = true;
	args.bGpuAccel = false;
	args.bParallel = true;
	Run(&args);
	return 0;
#endif
}
