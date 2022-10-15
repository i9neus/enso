#pragma once

#include "DeviceObjectTestsImpl.cuh"

using namespace Cuda;

namespace Tests
{
	TEST_CLASS(DeviceObjectTests), public SuiteBase
	{
	public:
		EXTERNAL_TEST_METHOD(DeviceObjectTestsImpl, ConstructDestruct)
		EXTERNAL_TEST_METHOD(DeviceObjectTestsImpl, Cast)
	};
}