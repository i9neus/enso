#pragma once

#include "CppUnitTest.h"
#include "generic\StdIncludes.h"
#include "generic\Math.h"
#include "kernels\math\CudaMath.cuh"
#include "kernels\CudaSampler.cuh"
#include "generic\StringUtils.h"

#include <random>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Cuda;

namespace Tests
{
	class SuiteBase
	{
	protected:
		std::mt19937 m_mt;
		std::uniform_real_distribution<> m_rng;

	protected:
		SuiteBase() :
			m_mt(0),
			m_rng(0.0f, 1.0f)
		{}

		void ReseedRNG(const uint seed)
		{
			m_mt = std::mt19937(seed);
		}

		inline vec3 RandVec3(const float rangeLow, const float rangeHigh)
		{
			return vec3(mix(rangeLow, rangeHigh, m_rng(m_mt)), mix(rangeLow, rangeHigh, m_rng(m_mt)), mix(rangeLow, rangeHigh, m_rng(m_mt)));
		}

		inline vec3 RandNormVec3()
		{
			return SampleUnitSphere(vec2(m_rng(m_mt), m_rng(m_mt)));
		}

		inline float Rand()
		{
			return  m_rng(m_mt);
		}
	};
	
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