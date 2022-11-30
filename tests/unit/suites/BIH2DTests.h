#pragma once

#include "BIH2DTestsImpl.cuh"

using namespace Enso;

namespace Tests
{
	TEST_CLASS(BIH2DTests), public SuiteBase
	{
	public:
		EXTERNAL_TEST_METHOD(CudaBIH2DTestsImpl, BuildSimpleGeometry)
		EXTERNAL_TEST_METHOD(CudaBIH2DTestsImpl, PointTestSimpleGeometry)
		EXTERNAL_TEST_METHOD(CudaBIH2DTestsImpl, RayTestSimpleGeometry)
		EXTERNAL_TEST_METHOD(CudaBIH2DTestsImpl, RayTestRandomGeometry)
	};
}