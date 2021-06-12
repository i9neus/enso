#pragma once

#include "math/CudaMath.cuh"

namespace Cuda
{
	#define kSpecularPdf  65000.0f;

	#define kRayAlive 1

	struct RayBasic
	{
	public:
		vec3 o, d;

		__device__ RayBasic() = default;
		__device__ RayBasic(const vec3& o_, const vec3& d_) : o(o_), d(d_) {}
		__device__ inline vec3 PointAt(const float& t) const { return o + d * t; }		
	};

	__device__ inline RayBasic RayToObjectSpace(const RayBasic& world, const BidirectionalTransform& bdt) 
	{
		RayBasic object;
		object.o = world.o - bdt.trans;
		object.d = world.d + object.o;
		object.o = bdt.fwd * object.o;
		object.d = (bdt.fwd * object.d) - object.o;
		return object;
	}

	__device__ inline vec3 PointToObjectSpace(const vec3& p, const BidirectionalTransform& bdt) { return bdt.fwd * (p - bdt.trans); }
	__device__ inline vec3 PointToWorldSpace(const vec3& p, const BidirectionalTransform& bdt) { return (bdt.inv * p) + bdt.trans; }
	__device__ inline vec3 NormalToWorldSpace(const vec3& n, const BidirectionalTransform& bdt) { return bdt.nInv * n; }

	struct CompressedRay
	{
		__host__ __device__ CompressedRay() : flags(0) {}

		RayBasic		od;			// 24 bytes
		vec3			weight;		// 12 bytes
		half			pdf;		// 2 bytes
		half			lambda;		// 2 bytes
		struct
		{
			ushort		x;			// 2 bytes
			ushort		y;			// 2 bytes
		}
		viewport;
		uchar	flags;				// 1 byte
		uchar	depth;				// 1 byte
		uint	sampleIdx;			// 4 bytes

		__device__ void SetAlive() { flags |= kRayAlive; }
		__device__ void Kill() { flags &= ~kRayAlive; }
		__device__ bool IsAlive() { return flags & kRayAlive; }
		__device__ ivec2 ViewportPos() const { return ivec2(viewport.x, viewport.y); }
	};

	// The "full fat" ray objects that most methods will refer to
	struct Ray
	{				
		RayBasic	od;		
		float   tNear;             // The parameterised intersection along the ray, defined in cartesian coordinates as o + d * tNear		
		vec3	weight;            // The weight/throughput of the ray depending on context
		float   pdf;               // The value of a probability density function incident to the intersection, depending on context
		float   lambda;            // The wavelength of the ray used by the spectral integrator		

		Ray() = default;
		__device__ Ray(const CompressedRay & comp) :
			od(comp.od),
			tNear(FLT_MAX),
			pdf(comp.pdf),
			lambda(lambda),
			weight(comp.weight)
		{
		}

		__device__ vec3 HitPoint() const { return od.o + od.d * tNear; }
	};
}
