#pragma once

#include "CudaVectorTestsImpl.cuh"

using namespace Cuda;

namespace Tests
{
	TEST_CLASS(CudaVectorTests), public SuiteBase
	{
	public:
		EXTERNAL_TEST_METHOD(CudaVectorTestsImpl, ConstructDestruct)
		EXTERNAL_TEST_METHOD(CudaVectorTestsImpl, Resize)
		//EXTERNAL_TEST_METHOD(CudaVectorTestsImpl, Reserve)
		EXTERNAL_TEST_METHOD(CudaVectorTestsImpl, Synchronise)
	};
}