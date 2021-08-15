#pragma once

#include "CudaSampler.cuh"

namespace Cuda
{
	using RNG = PseudoRNG;

	struct RenderCtx
	{
		__device__ __forceinline__ RenderCtx(CompressedRay& compressed) :
			emplacedRay(compressed),
			depth(compressed.depth),
			rng(compressed)
		{}

		uchar			depth;
		RNG				rng;
		CompressedRay&  emplacedRay;

		__device__ __forceinline__ void ResetRay() { emplacedRay.flags = 0; }

		__device__ __forceinline__ void EmplaceIndirectSample(const RayBasic& od, const vec3& weight, const uchar& flags)
		{
			emplacedRay.od = od;
			emplacedRay.weight = weight;
			emplacedRay.depth++;
			emplacedRay.flags = kRayIndirectSample | (emplacedRay.flags & kRayPersistentFlags) | flags;
		}

		__device__ __forceinline__ void EmplaceDirectSample(const RayBasic& od, const vec3& weight, const float& pdf, const ushort& lightId, const uchar& flags)
		{
			emplacedRay.od = od;
			emplacedRay.weight = weight;
			emplacedRay.lightId = lightId;
			emplacedRay.pdf = pdf;
			emplacedRay.depth++;
			emplacedRay.flags = flags | (emplacedRay.flags & kRayPersistentFlags);
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
		vec3		debug;

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