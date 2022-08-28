#pragma once

#include "JsonUtils.h"
#include <stdio.h>
#include <unordered_map>
#include <functional>
#include <atomic>

enum JobState : int
{
	kJobInvalidState = 0,
	kJobIdle = 1,
	kJobDispatched = 2,
	kJobRunning = 4,
	kJobCompleted = 8,
	kJobAborting = 16,
	kJobActive = kJobDispatched | kJobRunning
};

class Job
{	
	Job() noexcept : state(kJobIdle) {}

	std::function<bool(Job&)>								onDispatch;
	std::function<bool(Json::Node&, Job&)>					onPoll;
	std::atomic<int>										state;
	Json::Document											json;
};

using JobMap = std::unordered_map<std::string, Job&>;

class JobManager
{
public:
	friend class JobFactory;

private:
	
};

class JobFactory
{
public:
	JobFactory() = default;

	template<typename HostType, typename DispatchLambda, typename PollLambda>
	void Register(Job& job, const std::string& name, DispatchLambda onDispatch, PollLambda onPoll)
	{
		m_jobMap.emplace(name, job);
		job.onDispatch = std::bind(onDispatch, this, std::placeholders::_1);
		job.onPoll = std::bind(onPoll, this, std::placeholders::_1, std::placeholders::_2);
	}

private:
	JobMap	m_jobMap;
};

