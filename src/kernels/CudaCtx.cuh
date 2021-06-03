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
			float		   wallTime;
			int			   frameIdx;
			uchar          depth;
			PCG            pcg;
		};

		struct HitCtx
		{
			Hit     hit;
			vec2	uv;                // UV parameterisation coordinate at the intersected surface
			bool	backfacing;        // Whether the intersection with a forward- or backward-facing surface
			float   kickoff;           // The degree to which extant rays should be displaced from the surface to prevent self-intersection

			__device__ void Set(const Hit& hit_, bool back, const vec2& uv_, const float kick)
			{
				hit = hit_;
				backfacing = back;
				uv = uv_;
				kickoff = kick;
			}
		};
	}
}