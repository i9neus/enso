#pragma once

#include "math/CudaMath.cuh"
#include "CudaSampler.cuh"
#include "CudaHash.cuh"

//#define kPNGSampler

namespace Cuda
{
	namespace Device
	{
		struct RenderCtx
		{
			__device__ RenderCtx(const ivec2& viewPos, const ivec2& viewDims, const float& wall, const int& sample, const uint& depth_) :
				viewportPos(viewPos),
				viewportDims(viewDims),
				wallTime(wall),
				sampleIdx(sample),
				haltonSeed(HashOf(uint(depth) + 9871251u, uint(viewPos.x), uint(viewPos.y))),
				depth(depth_),
				pcg(HashCombine(HashOf(uint(sampleIdx)), haltonSeed))
			{
				emplacedRay.flags = 0;
			}
			
			const ivec2		viewportPos;
			const ivec2&	viewportDims;
			const uchar     depth;
			const float		wallTime;
			const int		sampleIdx;
			const uint		haltonSeed;
			PCG				pcg;

			CompressedRay  emplacedRay;

#ifdef kPNGSampler
			__device__ inline vec4 Rand4() { return pcg.Rand(); }
			__device__ inline vec3 Rand3() { return pcg.Rand().xyz; }
			__device__ inline vec2 Rand2() { return pcg.Rand().xy; }
			__device__ inline float Rand() { return pcg.Rand().x; }
#else
			__device__ inline vec4 Rand4() 
			{ 
				const vec4 halton(HaltonBase2(haltonSeed + sampleIdx), HaltonBase3(haltonSeed + sampleIdx), HaltonBase5(haltonSeed + sampleIdx), HaltonBase7(haltonSeed + sampleIdx));
				return fmod(halton + pcg.Rand(haltonSeed), 1.0f);
			}
			__device__ inline vec3 Rand3() 
			{ 
				const vec3 halton(HaltonBase2(haltonSeed + sampleIdx), HaltonBase3(haltonSeed + sampleIdx), HaltonBase5(haltonSeed + sampleIdx));
				return fmod(halton + pcg.Rand(haltonSeed).xyz, 1.0f);
			}
			__device__ inline vec2 Rand2()
			{
				const vec2 halton(HaltonBase2(haltonSeed + sampleIdx), HaltonBase3(haltonSeed + sampleIdx));
				return fmod(halton + pcg.Rand(haltonSeed).xy, 1.0f);
			}
			__device__ inline float Rand1()
			{
				return fmodf(HaltonBase2(haltonSeed + sampleIdx) + pcg.Rand(haltonSeed).x, 1.0f);
			}
#endif

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
				emplacedRay.sampleIdx = sampleIdx;

				emplacedRay.flags |= kRayAlive;
			}
		};

		struct HitPoint
		{
		public:
			vec3 p, n;

			__device__ HitPoint() = default;
			__device__ HitPoint(const vec3& p_, const vec3& n_) : p(p_), n(n_) {}
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

			__device__ inline vec3 ExtantOrigin() const { return hit.p + hit.n * kickoff; }

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