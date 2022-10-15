#pragma once

#include "CppUnitTest.h"
#include "generic\WindowsHeaders.h"
#include "kernels\CudaAsset.cuh"
#include "kernels\math\CudaMath.cuh"
#include "kernels\CudaSampler.cuh"
#include "generic\StringUtils.h"
#include <functional>

#include <random>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Cuda;

#define EXTERNAL_TEST_METHOD(ClassName, FuncName) \
	TEST_METHOD(FuncName) \
	{ \
		ClassName instance; \
		void(*ptr)(const wchar_t*) = Logger::WriteMessage; \
		instance.SetDebugFunctor(std::function<void(const wchar_t*)>(ptr)); \
		instance.FuncName(); \
	}

namespace Tests
{
	class SuiteBase
	{
	public:
		void SetDebugFunctor(std::function<void(const wchar_t*)>& functor) { m_writeMessageFunctor = functor; }

	protected:
		std::mt19937 m_mt;
		std::uniform_real_distribution<> m_realRng;
		std::uniform_int_distribution<> m_intRng;

	protected:
		SuiteBase() :
			m_mt(0),
			m_realRng(0.0f, 1.0f)
		{}

		void ReseedRNG(const uint seed)
		{
			m_mt = std::mt19937(seed); 
		}

		inline float RandFlt()
		{
			return  m_realRng(m_mt);
		}

		inline float RandFlt(const float rangeLow, const float rangeHigh)
		{
			return mix(rangeLow, rangeHigh, m_realRng(m_mt));
		}

		inline int RandInt()
		{
			return m_intRng(m_mt);
		}

		inline int RandInt(const int rangeLow, const int rangeHigh)
		{
			Assert::IsTrue(rangeLow < rangeHigh, L"Invalid range");
			return rangeLow + m_intRng(m_mt) % (rangeHigh - rangeLow);
		}

		inline ivec2 RandIVec2(const int rangeLow, const int rangeHigh)
		{
			return ivec2(RandInt(rangeLow, rangeHigh), RandInt(rangeLow, rangeHigh));
		}

		inline ivec3 RandIVec3(const int rangeLow, const int rangeHigh)
		{
			return ivec3(RandInt(rangeLow, rangeHigh), RandInt(rangeLow, rangeHigh), RandInt(rangeLow, rangeHigh));
		}

		inline ivec4 RandIVec4(const int rangeLow, const int rangeHigh)
		{
			return ivec4(RandInt(rangeLow, rangeHigh), RandInt(rangeLow, rangeHigh), RandInt(rangeLow, rangeHigh), RandInt(rangeLow, rangeHigh));
		}

		inline vec2 RandIVec2(const float rangeLow, const float rangeHigh)
		{
			return vec2(mix(rangeLow, rangeHigh, m_realRng(m_mt)), mix(rangeLow, rangeHigh, m_realRng(m_mt)));
		}

		inline vec2 RandVec2(const float rangeLow, const float rangeHigh)
		{
			return vec2(RandFlt(rangeLow, rangeHigh), RandFlt(rangeLow, rangeHigh));
		}

		inline vec3 RandVec3(const float rangeLow, const float rangeHigh)
		{
			return vec3(RandFlt(rangeLow, rangeHigh), RandFlt(rangeLow, rangeHigh), RandFlt(rangeLow, rangeHigh));
		}

		inline vec4 RandVec4(const float rangeLow, const float rangeHigh)
		{
			return vec4(RandFlt(rangeLow, rangeHigh), RandFlt(rangeLow, rangeHigh), RandFlt(rangeLow, rangeHigh), RandFlt(rangeLow, rangeHigh));
		}

		inline vec3 RandNormVec3()
		{
			return SampleUnitSphere(vec2(m_realRng(m_mt), m_realRng(m_mt)));
		}

		void DebugMessage(const std::string& msg)
		{
			if (m_writeMessageFunctor)
			{
				m_writeMessageFunctor(Widen(msg).c_str());
			}
			else
			{
				Logger::WriteMessage(Widen(msg).c_str());
			}
		}

	protected:
		std::function<void(const wchar_t*)>	m_writeMessageFunctor;
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