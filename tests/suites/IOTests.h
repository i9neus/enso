#pragma once

#include "SuiteBase.h"
#include <random>

#include <kernels/lightprobes/CudaLightProbeDataTransform.cuh>

namespace Tests
{
	TEST_CLASS(ProbeDataFransformTests), public SuiteBase
	{
	public:
		void CheckForwardBackwardMap(const LightProbeGridParams& params)
		{
			std::vector<vec3> referenceData(params.numProbes * params.coefficientsPerProbe);

			// Create a light probe grid with random data
			for (int x = 0, coeffIdx = 0; x < params.gridDensity.x; ++x)
			{
				for (int y = 0; y < params.gridDensity.x; ++y)
				{
					for (int z = 0; z < params.gridDensity.x; ++z, coeffIdx += params.coefficientsPerProbe)
					{
						referenceData[0] = RandVec3(0.0f, 1.0f);
						for (int sh = 1; sh < 4; ++sh)
						{
							referenceData[coeffIdx + sh] = RandVec3(-1.0f, 1.0f);
						}
						referenceData[coeffIdx + 4] = kZero;
						referenceData[coeffIdx + 4].x = Rand();
					}
				}
			}

			// Make a copy
			std::vector<vec3> inputData = referenceData;
			std::vector<vec3> outputData(referenceData.size());

			// Forward and backward transform the data
			LightProbeDataTransform transform(params);
			transform.Forward(inputData, outputData);
			transform.Inverse(outputData, inputData);

			// Verify the data integrity
			constexpr float kErrorThreshold = 1e-4f;
			float sumError = 0.0;
			for (int x = 0, coeffIdx = 0; x < params.gridDensity.x; ++x)
			{
				for (int y = 0; y < params.gridDensity.x; ++y)
				{
					for (int z = 0; z < params.gridDensity.x; ++z)
					{
						for (int sh = 0; sh < params.coefficientsPerProbe; ++sh, ++coeffIdx)
						{
							for (int c = 0; c < 3; ++c)
							{
								float error = std::abs(inputData[coeffIdx][c] - referenceData[coeffIdx][c]);
								sumError += error;
								Logger::WriteMessage(tfm::format("%i, %i, %i, %i, %i: %.2f -> %.2f\n", x, y, z, sh, c, referenceData[coeffIdx][c], inputData[coeffIdx][c]).c_str());
								/*Assert::IsTrue(std::abs(inputData[coeffIdx][c] - referenceData[coeffIdx][c]) < kErrorEpsilon,
									Widen(tfm::format("Element ", a.format(), c.format())).c_str())*/
							}
						}
					}
				}
			}

			Assert::IsTrue(sumError < kErrorThreshold, Widen(tfm::format("Error %.5f exceeded threshold of %.5f", sumError, kErrorThreshold)).c_str());
		}

		LightProbeGridParams DefaultGridParams() const
		{
			LightProbeGridParams params;

			params.gridDensity = ivec3(2, 2, 2);
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
		
			CheckForwardBackwardMap(params);
		}

		TEST_METHOD(CheckPosSwizzle)
		{
			LightProbeGridParams params = DefaultGridParams();

			params.posSwizzle = kYXZ;

			CheckForwardBackwardMap(params);
		}

		TEST_METHOD(CheckSHSwizzle)
		{
			LightProbeGridParams params = DefaultGridParams();

			params.shSwizzle = kXYZ;

			CheckForwardBackwardMap(params);
		}

		TEST_METHOD(CheckPositionInvert)
		{
			LightProbeGridParams params = DefaultGridParams();
		
			params.posInvertX = true;
			params.posInvertY = true;
			params.posInvertZ = true;

			CheckForwardBackwardMap(params);
		}

		TEST_METHOD(CheckSHInvert)
		{
			LightProbeGridParams params = DefaultGridParams();

			params.shInvertX = true;
			params.shInvertY = true;
			params.shInvertZ = true;

			CheckForwardBackwardMap(params);
		}
	};
}