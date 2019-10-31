/*
 * main.cpp
 *
 *  Created on: Mar 1, 2019
 *      Author: Alena Kazakova
 */

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/mat.inl.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stddef.h>
#include <chrono>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <memory>

#include "lane_following/color_grad_thresh.h"
#include "lane_following/cuda_color_grad_thresh.h"
#include "lane_following/cuda_warp.h"
#include "lane_following/debug.h"
#include "lane_following/find_lanes.h"
#include "lane_following/lane_base.h"
#include "lane_following/thread_base.h"
#include "lane_following/thread_manager.h"
#include "lane_following/warp.h"

void CudaInit() {
	cv::cuda::GpuMat gpu_img;
	gpu_img.create(1, 1, CV_8U);
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
	auto init_time = std::chrono::high_resolution_clock::now();

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
		std::signal(SIGINT,
				ThreadManager<CudaWarp, CudaColorGradThresh, FindLanes>::stSignalHandler);
		threadManagerCuda->AddMsg(msg);
		threadManagerCuda->WaitForThread();
	}
	if (threadManager) {
		std::signal(SIGINT,
				ThreadManager<Warp, ColorGradThresh, FindLanes>::stSignalHandler);
		threadManager->AddMsg(msg);
		threadManager->WaitForThread();
	}

	auto init_duration = std::chrono::duration_cast<std::chrono::microseconds>(
			start_time - init_time);
	auto end_time = std::chrono::high_resolution_clock::now();
	auto exec_duration = std::chrono::duration_cast<std::chrono::microseconds>(
			end_time - start_time);
#if DEBUG_ZONE_TEST
	//std::cout << std::left << std::setw(8)
	//		<< exec_duration.count() / args->maxFrameCnt;
	if (threadManagerCuda) {
		std::cout << std::left << std::setw(8) << threadManagerCuda->getAvgDuration();
	}
	if (threadManager) {
		std::cout << std::left << std::setw(8) << threadManager->getAvgDuration();
	}
#elif DEBUG_ZONE_TEST_SPEED
	std::cout << "<" << (args->bParallel ? "Parallel" : "")
			<< (args->bGpuAccel ? "GpuAccel" : "")
			<< (args->pipelineInstNum > 1 ? "Pipeline" : "Sequential") << ">"
			<< std::endl;
	std::cout << std::left << std::setw(8) << "Time" << std::setw(8) << "Speed"
			<< std::endl;
	if (threadManagerCuda) {
		std::cout << std::left << std::setw(8)
				<< threadManagerCuda->getAvgDuration() << std::setw(8)
				<< threadManagerCuda->getAvgSpeed() << std::endl;
	}
	if (threadManager) {
		std::cout << std::left << std::setw(8)
				<< threadManager->getAvgDuration() << std::setw(8)
				<< threadManager->getAvgSpeed() << std::endl;
	}
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
	std::cout << std::left << std::setw(20) << "Initialization time" << std::setw(12)
			<< init_duration.count() << " usec" << std::endl;
	std::cout << std::left << std::setw(20) << "Total execution time" << std::setw(12)
			<< exec_duration.count() << " usec" << std::endl;
	std::cout << std::left << std::setw(20) << "Avg execution time" << std::setw(12)
			<< exec_duration.count() / args->maxFrameCnt << " usec" << std::endl
			<< std::endl;
	if (threadManagerCuda) {
		threadManagerCuda->PrintAvgFuncDurations();
	}
	if (threadManager) {
		threadManager->PrintAvgFuncDurations();
	}
#endif
	delete threadManagerCuda;
	delete threadManager;
	ThreadBase::ResetThreadRegister();

}
void TestHelper(tm_args *args) {
#if DEBUG_ZONE_TEST
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
#elif DEBUG_ZONE_TEST_SPEED
	args->threadPoolSize = 8;
	args->pipelineInstNum = 1;
	Run(args);
	args->pipelineInstNum = 8;
	Run(args);
#endif
}

void Test() {
	tm_args args;
	args.videoFile = "raw_video.avi";
	args.maxFrameCnt = 100;
	args.bVerbose = false;

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
#if DEBUG_ZONE_ROS
	ros::init(argc, argv, "lane_following_node",
	ros::init_options::NoSigintHandler);
#endif

#if DEBUG_ZONE_TEST || DEBUG_ZONE_TEST_SPEED
	Test();
#else
	tm_args args;
	args.videoFile = "raw_video.avi";
	args.threadPoolSize = 8;
	args.pipelineInstNum = 4;
	args.maxFrameCnt = 100;
	args.speed = 3000;
	args.delay = 0;
	args.bVerbose = true;
	args.bGpuAccel = true;
	args.bParallel = true;
	Run(&args);
	return 0;
#endif
}
