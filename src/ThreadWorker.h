#ifndef SRC_THREADWORKER_H_
#define SRC_THREADWORKER_H_

#include <memory>

#include "ThreadBase.h"

class ThreadWorker: public ThreadBase {
public:
	enum TASK_MSG {
		TASK_MSG_RUN_WARP = THREAD_MSG::THREAD_MSG_SYS_MAX,
		TASK_MSG_RUN_COLOR_GRAD_THRESH,
		TASK_MSG_RUN_FIND_LANES,
		TASK_MSG_COMPLETE_WARP,
		TASK_MSG_COMPLETE_COLOR_GRAD_THRESH,
		TASK_MSG_COMPLETE_FIND_LANES,
		TASK_MSG_EXISTS_FIND_LANES
	};
	virtual void ProcessMsg(std::shared_ptr<ThreadMsg> &msg) override;
	void WarpRun(std::shared_ptr<ThreadMsg> &msg);
	void ColorGradThreshRun(std::shared_ptr<ThreadMsg> &msg);
	void FindLanesRun(std::shared_ptr<ThreadMsg> &msg);
	virtual bool PreWorkInit() override;
	virtual bool PostWorkDeinit() override;
};

#endif /* SRC_THREADWORKER_H_ */
