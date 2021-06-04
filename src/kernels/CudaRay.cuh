#pragma once

#include "math/CudaMath.cuh"

namespace Cuda
{
	#define kSpecularPdf  65000.0f;

	#define kRayAlive 1

	using RayBasic = PosDir;
	using HitPoint = PosDir;

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
		uchar   depth;

		Ray() = default;
		__device__ Ray(const CompressedRay & comp) :
			od(comp.od),
			tNear(FLT_MAX),
			pdf(comp.pdf),
			lambda(lambda),
			weight(comp.weight),
			depth(comp.depth)
		{
		}
	};
}
