#pragma once

#include "SuiteBase.h"
#include <random>

#include <kernels/lightprobes/CudaLightProbeDataTransform.cuh>

namespace Tests
{
	TEST_CLASS(ProbeDataFransformTests), public SuiteBase
	{
	public:
		void SeedRandomData(std::vector<vec3>& data, const int coeffIdx)
		{
			// Seed the SH values with random coefficients 
			data[coeffIdx + 0] = RandVec3(0.0f, 1.0f);
			for (int sh = 1; sh < 4; ++sh)
			{
				data[coeffIdx + sh] = RandVec3(-1.0f, 1.0f);
				}
			data[coeffIdx + 4] = kZero;
			data[coeffIdx + 4].x = Rand();
		}

		void SeedDiagnosticData(std::vector<vec3>& data, const ivec3& probePos, const int coeffIdx)
		{
			vec3 v(probePos);
			data[coeffIdx + 0] = v;
			data[coeffIdx + 1] = kZero;
			data[coeffIdx + 2] = kZero;
			data[coeffIdx + 3] = kZero;
			data[coeffIdx + 4] = kZero;
		}

		void CheckForwardBackwardMap(const LightProbeGridParams& params, const bool forwardOnly)
		{
			std::vector<vec3> referenceData(params.numProbes * params.coefficientsPerProbe);

			// Create a light probe grid with random data
			for (int x = 0, coeffIdx = 0; x < params.gridDensity.x; ++x)
			{
				for (int y = 0; y < params.gridDensity.y; ++y)
				{
					for (int z = 0; z < params.gridDensity.z; ++z, coeffIdx += params.coefficientsPerProbe)
					{
						SeedRandomData(referenceData, coeffIdx);
						//SeedDiagnosticData(referenceData, ivec3(x, y, z), coeffIdx);
					}
				}
			}

			// Make a copy
			std::vector<vec3> inputData = referenceData;
			std::vector<vec3> outputData(referenceData.size());

			// Forward and backward transform the data
			LightProbeDataTransform transform(params);
			transform.Forward(inputData, outputData);
			
			if (!forwardOnly)
			{
				std::swap(inputData, outputData);
				transform.Inverse(inputData, outputData);
			}

			// Verify the data integrity
			constexpr float kErrorThreshold = 1e-3f;
			float sumError = 0.0;
			for (int x = 0, coeffIdx = 0; x < params.gridDensity.x; ++x)
			{
				for (int y = 0; y < params.gridDensity.y; ++y)
				{
					for (int z = 0; z < params.gridDensity.z; ++z)
					{
						
						for (int sh = 0; sh < params.coefficientsPerProbe; ++sh, ++coeffIdx)
						{
							//Logger::WriteMessage(tfm::format("%i, %i, %i, -> %f, %f, %f\n", x, y, z, (referenceData[coeffIdx][0]), (outputData[coeffIdx][1]), (outputData[coeffIdx][2])).c_str()); 
							for (int c = 0; c < 3; ++c)
							{
								float error = std::abs(outputData[coeffIdx][c] - referenceData[coeffIdx][c]);
								sumError += error;								
								/*Assert::IsTrue(std::abs(outputData[coeffIdx][c] - referenceData[coeffIdx][c]) < kErrorEpsilon,
									Widen(tfm::format("Element ", a.format(), c.format())).c_str())*/
							}
						}
					}
				}
			}

			// Forward only test should result a large error
			if (forwardOnly)
			{
				Assert::IsFalse(sumError < kErrorThreshold, Widen(tfm::format("Error %.5f is close to zero.", sumError, kErrorThreshold)).c_str());
			}
			else
			{
				Assert::IsTrue(sumError < kErrorThreshold, Widen(tfm::format("Error %.5f exceeded threshold of %.5f", sumError, kErrorThreshold)).c_str());
			}
		}

		LightProbeGridParams DefaultGridParams(const ivec3 gridDensity = ivec3(10, 10, 10)) const
		{
			LightProbeGridParams params;

			params.gridDensity = gridDensity;
			params.shOrder = 1;
			params.posSwizzle = kXYZ;
			params.shSwizzle = kXYZ;
			params.posInvertX = params.posInvertY = params.posInvertZ = false;
			params.shInvertX = params.shInvertY = params.shInvertZ = false;
			params.Prepare();
			
			return params;
		}

		TEST_METHOD(CheckIdentityTransform)
		{
			LightProbeGridParams params = DefaultGridParams();
		
			CheckForwardBackwardMap(params, false);
		}

		TEST_METHOD(CheckPosSwizzle)
		{
			LightProbeGridParams params = DefaultGridParams();

			params.posSwizzle = kYXZ;

			CheckForwardBackwardMap(params, false);
		}

		TEST_METHOD(CheckSHSwizzle)
		{
			LightProbeGridParams params = DefaultGridParams();

			params.shSwizzle = kXYZ;

			CheckForwardBackwardMap(params, false);
		}

		TEST_METHOD(CheckPosInvert)
		{
			LightProbeGridParams params = DefaultGridParams();
		
			params.posInvertX = true;
			params.posInvertY = true;
			params.posInvertZ = true;

			CheckForwardBackwardMap(params, false);
		}

		TEST_METHOD(CheckSHInvert)
		{
			LightProbeGridParams params = DefaultGridParams();

			params.shInvertX = true;
			params.shInvertY = true;
			params.shInvertZ = true;

			CheckForwardBackwardMap(params, false);
		}

		TEST_METHOD(CheckNonSquareGrid)
		{
			std::uniform_int_distribution<> rng;

			LightProbeGridParams params = DefaultGridParams(ivec3(2 + rng(m_mt) % 18, 2 + rng(m_mt) % 18, 2 + rng(m_mt) % 18));

			CheckForwardBackwardMap(params, false);
		}

		void CheckRandomParamsImpl(std::uniform_int_distribution<>& rng, const bool forwardOnly)
		{
			LightProbeGridParams params;

			params.gridDensity = ivec3(2 + rng(m_mt) % 18, 2 + rng(m_mt) % 18, 2 + rng(m_mt) % 18);
			params.shOrder = 1;
			params.posSwizzle = rng(m_mt) % 6;
			params.shSwizzle = rng(m_mt) % 6;
			params.posInvertX = bool(rng(m_mt) % 2);
			params.posInvertY = bool(rng(m_mt) % 2);
			params.posInvertZ = bool(rng(m_mt) % 2);
			params.shInvertX = bool(rng(m_mt) % 2);
			params.shInvertY = bool(rng(m_mt) % 2);
			params.shInvertZ = bool(rng(m_mt) % 2);
			params.Prepare();

			CheckForwardBackwardMap(params, false);
		}

		TEST_METHOD(CheckForwardOnly)
		{
			CheckRandomParamsImpl(std::uniform_int_distribution<>(), true);
		}

		TEST_METHOD(CheckRandomParams)
		{
			std::uniform_int_distribution<> rng;
			
			constexpr int kNumRuns = 10;
			for (int runIdx = 0; runIdx < kNumRuns; ++runIdx)
			{
				CheckRandomParamsImpl(rng, false);
			}
		}
	};
}