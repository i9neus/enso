#pragma once

#include "math/CudaMath.cuh"
#include "CudaSampler.cuh"
#include "CudaHash.cuh"
#include "CudaRay.cuh"

#define kPNGSampler

namespace Cuda
{
	struct RenderCtx
	{		
		__device__ __forceinline__ RenderCtx(CompressedRay& compressed, const ivec2& viewDims) :
			emplacedRay(compressed),
			viewportPos(compressed.ViewportPos()),
			viewportDims(viewDims),
			sampleIdx(compressed.sampleIdx),
			haltonSeed(HashOf(uint(depth) + 9871251u, uint(viewportPos.x), uint(viewportPos.y))),
			depth(compressed.depth)
#ifdef kPNGSampler
			,pcg(HashCombine(HashOf(uint(sampleIdx)), haltonSeed))
#endif
		{
		}

		ivec2			viewportPos;
		const ivec2&	viewportDims;
		uchar			depth;
		int				sampleIdx;
		const uint		haltonSeed;
		PCG				pcg;

		CompressedRay&  emplacedRay;

#ifdef kPNGSampler
		template<int B0, int B1, int B2, int B3>
		__device__ __forceinline__ vec4 Rand() { return pcg.Rand(); }
		template<int B0, int B1, int B2>
		__device__ __forceinline__ vec3 Rand() { return pcg.Rand().xyz; }
		template<int B0, int B1>
		__device__ __forceinline__ vec2 Rand() { return pcg.Rand().xy; }
		template<int B0>
		__device__ __forceinline__ float Ran() { return pcg.Rand().x; }
#else
		template<int B0, int B1, int B2, int B3>
		__device__ __forceinline__ vec4 Rand() const
		{
			return Halton<vec4, B0, B1, B2, B3>(haltonSeed + sampleIdx);
		}
		template<int B0, int B1, int B2>
		__device__ __forceinline__ vec3 Rand() const 
		{
			return Halton<vec3, B0, B1, B2>(haltonSeed + sampleIdx);
		}
		template<int B0, int B1>
		__device__ __forceinline__ vec2 Rand() const 
		{
			return Halton<vec2, B0, B1>(haltonSeed + sampleIdx);
		}
		template<int B0>
		__device__ __forceinline__ float Rand() const
		{
			return HaltonBase<B0>(haltonSeed + sampleIdx);
		}
#endif
		__device__ __forceinline__ void ResetRay() { emplacedRay.flags = 0; }

		__device__ __forceinline__ void EmplaceIndirectSample(const RayBasic& od, const vec3& weight)
		{
			emplacedRay.od = od;
			emplacedRay.weight = weight;
			emplacedRay.depth++;
			emplacedRay.flags = kRayIndirectSample;
		}

		__device__ __forceinline__ void EmplaceDirectSample(const RayBasic& od, const vec3& weight, const float& pdf, const ushort& lightId, const uchar& flags)
		{
			emplacedRay.od = od;
			emplacedRay.weight = weight;
			emplacedRay.lightId = lightId;
			emplacedRay.pdf = pdf;
			emplacedRay.depth++;
			emplacedRay.flags = flags;
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

		__device__ __forceinline__ vec3 ExtantOrigin() const { return hit.p + hit.n * kickoff; }

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