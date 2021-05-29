#pragma once

#include "math/CudaMath.cuh"
#include "CudaSampler.cuh"

namespace Cuda
{
	namespace Device
	{
		struct RenderCtx
		{
			ivec2          viewportPos;
			ivec2		   viewportDims;
			PCG            pcg;
		};

		struct HitCtx
		{
			vec3	n;                 // Normal at the intersected surface
			vec2	uv;                // UV parameterisation coordinate at the intersected surface
			bool	backfacing;        // Whether the intersection with a forward- or backward-facing surface
			float   kickoff;           // The degree to which extant rays should be displaced from the surface to prevent self-intersection

			__device__ void Set(const vec3& n_, bool back, const vec2& uv_, const float kick)
			{
				n = n_;
				backfacing = back;
				uv = uv_;
				kickoff = kick;
			}
		};
	}
}