#pragma once

#include "CudaBIH2DTestsImpl.cuh"

using namespace Cuda;

namespace Tests
{
	TEST_CLASS(CudaBIH2DTests), public SuiteBase
	{
	public:
		EXTERNAL_TEST_METHOD(CudaBIH2DTestsImpl, BuildSimpleGeometry)
		EXTERNAL_TEST_METHOD(CudaBIH2DTestsImpl, RuildRandomGeometry)
		EXTERNAL_TEST_METHOD(CudaBIH2DTestsImpl, PointTestSimpleGeometry)
		EXTERNAL_TEST_METHOD(CudaBIH2DTestsImpl, RayTestSimpleGeometry)
	};
}