#pragma once

#include "CudaSampler.cuh"

namespace Cuda
{
	using RNG = PseudoRNG;

	enum LightIDFlags : uchar { kNotALight = 0xff };

	struct RenderCtx
	{
		__device__ __forceinline__ RenderCtx(CompressedRay& compressed) :
			emplacedRay(&compressed),
			depth(compressed.depth),
			rng(compressed)
		{}

		uchar			depth;
		RNG				rng;
		CompressedRay*  emplacedRay;

		__device__ __forceinline__ void ResetRay() { emplacedRay->flags = 0; }

		__device__ __forceinline__ void EmplaceIndirectSample(const RayBasic& od, const vec3& weight, const uchar& flags)
		{
			auto& ray = emplacedRay[0];
			ray.od = od;
			ray.weight = weight;
			ray.depth++;
			ray.flags = kRayIndirectSample | (ray.flags & kRayPersistentFlags) | flags;
		}

		__device__ __forceinline__ void EmplaceDirectSample(const RayBasic& od, const vec3& weight, const float& pdf, const ushort& lightId, const uchar& flags, const Ray& parent)
		{
			auto& ray = emplacedRay[1];
			ray.od = od;
			ray.weight = weight;
			ray.lightId = lightId;
			ray.pdf = pdf;
			ray.depth = parent.depth + 1;
			ray.accumIdx = emplacedRay[0].accumIdx;
			ray.flags = flags | (parent.flags & kRayPersistentFlags);
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
		uchar		lightID;

		__device__ HitCtx() : isValid(false), lightID(kNotALight) {}

		__device__ __forceinline__ vec3 ExtantOrigin() const { return hit.p + hit.n * kickoff; }

		__device__ void Set(const HitPoint& hit_, bool back, const vec2& uv_, const float kick, const uchar ID)
		{
			hit = hit_;
			backfacing = back;
			uv = uv_;
			kickoff = kick;
			isValid = true;
			lightID = ID;
		}
	};
}