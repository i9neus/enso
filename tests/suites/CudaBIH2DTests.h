#pragma once

#include "SuiteBase.h"
#include <random>

#include <kernels/gi2d/CudaBIH2D.cuh>

using namespace Cuda;

namespace Tests
{	
	TEST_CLASS(BoundingIntervalHierarchy2DTests), public SuiteBase
	{
	public:
		TEST_METHOD(RandomPrimitives)
		{
			constexpr int kNumPrims = 1000;
			constexpr float kBoundSize = 1.0f;
			constexpr float kPrimSize = 0.01f;
			std::vector<BBox2f> primBBoxes(kNumPrims);
			std::vector<uint> primIdxs(kNumPrims);
			
			// Construct an array of random primitives
			for (int idx = 0; idx < primBBoxes.size(); ++idx)
			{
				vec2 p = RandVec2(-0.5f, 0.5f) * kBoundSize;
				vec2 dp = RandVec2(0.1, 1.0) * kBoundSize * kPrimSize;
				primBBoxes[idx] = BBox2f(p - dp, p + dp);
				primIdxs[idx] = idx;
			}

			AssetHandle<Host::BIH2D> bih = CreateAsset<Host::BIH2D>("bih");
			Host::BIH2DBuilder builder(*bih);
			
			builder.Build(primIdxs, [&primBBoxes](const uint idx) -> BBox2f& { Assert(idx < primBBoxes.size());  return primBBoxes[idx]; });

			bih.DestroyAsset();
		}		
	};
}