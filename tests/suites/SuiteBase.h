#pragma once

#include "CppUnitTest.h"
#include "generic\StdIncludes.h"
#include "generic\Math.h"
#include "kernels\math\CudaMath.cuh"
#include "kernels\CudaRay.cuh"
#include "kernels\tracables\CudaGenericIntersectors.cuh"
#include "generic\StringUtils.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Cuda;

namespace Tests
{
	class MatrixTestUtils
	{
	public:
		template<typename T>
		void TestMatIsEqual(const T& matBaseline, const T& matTest, const float kEpsilon)
		{
			for (int i = 0; i < T::kDims; i++)
			{
				for (int j = 0; j < T::kDims; j++)
				{
					Assert::IsTrue(std::abs(matBaseline[i][j] - matTest[i][j]) < kEpsilon,
						Widen(tfm::format("Elements are not equal: [%i, %i] %.10f should be %.10f", i, j, matTest[i][j], matBaseline[i][j])).c_str());
				}
			}
		}

		template<typename T>
		void TestVecIsEqual(const T& vecBaseline, const T& vecTest, const float kEpsilon, const char* name)
		{
			for (int i = 0; i < T::kDims; i++)
			{
				Assert::IsTrue(std::abs(vecBaseline[i] - vecTest[i]) < kEpsilon,
					Widen(tfm::format("Element %i of %s is not equal: %s should be %s", i, name, vecTest.format(), vecBaseline.format())).c_str());
			}
		}

		template<typename T>
		void CheckTypeSize(size_t targetSize, const char* name)
		{
			Assert::IsTrue(sizeof(T) == targetSize,
				Widen(tfm::format("%s is the wrong size: %i should be %i", name, sizeof(T), targetSize)).c_str());
		}
	};
}