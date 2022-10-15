#pragma once

#include "SuiteBase.h"

namespace Tests
{
	TEST_CLASS(CudaMathTests), MatrixTestUtils
	{
	public:
		TEST_METHOD(TestBidirTransformScale)
		{
			// Test that the matrix decomposition is correct for BidirectionalTransform object.

			const vec3 scale(0.183276f, 2.083f, 6.0385f);
			const vec3 rotate(1.5124f, -0.873651f, 0.58367f);
			const vec3 translate(0.0f);

			const BidirectionalTransform transform(rotate, translate, scale);

			//MatrixTestUtils(scale, transform.scale, 1e-7f, "scale");
		}

		/*TEST_METHOD(TestBidirObjectToWorld)
		{
			// Transform a hit point from object space into world space

			const BidirectionalTransform transform = CreateCompoundTransform(vec3(kHalfPi, 0.0f, 0.0f));

			const vec3 oObject(0.0f, 1.0f, 0.0f);
			const vec3 nObject(0.0f, 0.0f, 1.0f);
			const vec3 oWorld(0.0f, 0.0f, -1.0f);
			const vec3 nWorld(0.0f, 1.0f, 0.0f);

			HitPoint hitWorld = transform.HitToWorldSpace(HitPoint(oObject, nObject));

			TestVecIsEqual(hitWorld.o, oWorld, 1e-7f, "origin");
		}

		TEST_METHOD(TestBidirWorldToObjectToWorld)
		{
			// Transform a random vector and normal into object space and back out again

			const BidirectionalTransform transform = CreateCompoundTransform(vec3(-kHalfPi, 0.0f, 0.0f));

			const vec3 oWorld(10.3765f, 8.987291f, -2.579872f);
			const vec3 nWorld = normalize(vec3(-1.826f, 0.825176f, -0.127982376f));

			const HitPoint hitObject = ToObjectSpace( HitPoint(oWorld, nWorld));
			const HitPoint hitWorld = transform.HitToWorldSpace(hitObject);

			TestVecIsEqual(hitWorld.o, oWorld, 1e-6f, "origin");
			TestVecIsEqual(hitWorld.n, nWorld, 1e-6f, "normal");
		}*/
	};
}