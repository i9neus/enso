#pragma once

#include "VectorTestsImpl.cuh"

using namespace Enso;

namespace Tests
{
	TEST_CLASS(VectorTests), public SuiteBase
	{
	public:
		EXTERNAL_TEST_METHOD(VectorTestsImpl, ConstructDestruct)
		EXTERNAL_TEST_METHOD(VectorTestsImpl, Resize)
		//EXTERNAL_TEST_METHOD(VectorTestsImpl, Reserve)
		EXTERNAL_TEST_METHOD(VectorTestsImpl, Synchronise)
	};
}