#pragma once

#include "math/CudaMath.cuh"

namespace Cuda
{
	#define kSpecularPdf  65000.0f;

	#define kRayAlive 1

	using RayBasic = PosDir;
	using Hit = PosDir;

	struct CompressedRay
	{
		__host__ __device__ CompressedRay() : flags(0) {}

		RayBasic		od;
		vec3			weight;
		half			pdf;
		half			lambda;
		struct
		{
			short		x;
			short		y;
		}
		viewport;
		uchar	flags;
		uchar	depth;

		__device__ void SetAlive() { flags |= kRayAlive; }
		__device__ void Kill() { flags &= ~kRayAlive; }
		__device__ bool IsAlive() { return flags & kRayAlive; }
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
	};
}
