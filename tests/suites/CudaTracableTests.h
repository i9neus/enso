#pragma once

#include "SuiteBase.h"

namespace Tests
{
	TEST_CLASS(CudaTracableTests), MatrixTestUtils
	{
	public:
		TEST_METHOD(CheckBoxIntersector)
		{
			// Ray inside box. Should return 0.
			{
				RayBasic ray(vec3(0.0f, 0.0f, 0.0f), normalize(vec3(0.7561f, 1.8375f, -0.276745f)));

				const float t = Intersector::RayBox(ray, 0.5f);

				Assert::IsTrue(t == 0.0f, Widen(tfm::format("Ray origin inside box: intersector is %f, should be be 0.0", t)).c_str());
			}

			// Ray outside box and intersects
			{
				RayBasic ray(vec3(-1.0f, -1.0f, -1.0f), normalize(vec3(1.0f, 1.0f, 1.0f)));

				const float t = Intersector::RayBox(ray, 0.5f);
				const float tReference = std::sqrt(3.0f * 0.5f * 0.5f);
				constexpr float kEpsilon = 1e-6f;

				Assert::IsTrue(std::abs(t - tReference) < kEpsilon, Widen(tfm::format("Ray hit box: intersector is %f, should be be 0.0", t)).c_str());
			}

			// Ray outside box and does not intersect
			{
				RayBasic ray(vec3(-1.0f, 2.0f, -1.0f), normalize(vec3(1.0f, 1.0f, 1.0f)));

				const float t = Intersector::RayBox(ray, 0.5f);

				Assert::IsTrue(t == kNoIntersect, Widen(tfm::format("Ray didn't hit box: intersector is %f, should be be 0.0", t)).c_str());
			}
		}
	};
}