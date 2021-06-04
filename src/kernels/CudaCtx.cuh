#pragma once

#include "math/CudaMath.cuh"
#include "CudaSampler.cuh"
#include "CudaHash.cuh"

namespace Cuda
{
	namespace Device
	{
		struct RenderCtx
		{
			__device__ RenderCtx(const ivec2& viewPos, const ivec2& viewDims, const float& wall, const int& frame, const uint& depth) :
				viewportPos(viewPos),
				viewportDims(viewDims),
				wallTime(wall),
				frameIdx(frame),
				pcg(hashOf(uint(frameIdx), uint(depth), uint(viewPos.x), uint(viewPos.y)))
			{
				emplacedRay.flags = 0;
			}
			
			const ivec2    viewportPos;
			const ivec2&   viewportDims;
			const float	   wallTime;
			const int	   frameIdx;
			PCG            pcg;

			CompressedRay  emplacedRay;

			__device__ inline vec4 Rand4() { return pcg.Rand(); }
			__device__ inline vec3 Rand3() { return pcg.Rand().xyz; }
			__device__ inline vec2 Rand2() { return pcg.Rand().xy; }
			__device__ inline float Rand() { return pcg.Rand().x; }

			__device__ inline void EmplaceRay(const RayBasic& od, const vec3& weight, const float& pdf, const float& lambda, const uchar& flags, const uchar& depth)
			{
				emplacedRay.od = od;
				emplacedRay.weight = weight;
				emplacedRay.lambda = lambda;
				emplacedRay.pdf = pdf;
				emplacedRay.lambda = lambda;
				emplacedRay.flags = flags;
				emplacedRay.viewport.x = viewportPos.x;
				emplacedRay.viewport.y = viewportPos.y;
				emplacedRay.depth = depth + 1;

				emplacedRay.flags |= kRayAlive;
			}
		};

		struct HitCtx
		{
			HitPoint	hit;
			vec2		uv;                // UV parameterisation coordinate at the intersected surface
			bool		backfacing;        // Whether the intersection with a forward- or backward-facing surface
			float		kickoff;           // The degree to which extant rays should be displaced from the surface to prevent self-intersection
			bool		isValid;
			vec3        albedo;

			__device__ HitCtx() : isValid(false) {}

			__device__ inline vec3 ExtantOrigin() const { return hit.o + hit.n * kickoff; }

			__device__ void Set(const HitPoint& hit_, bool back, const vec2& uv_, const float kick)
			{
				hit = hit_;
				backfacing = back;
				uv = uv_;
				kickoff = kick;
				isValid = true;
			}
		};
	}
}