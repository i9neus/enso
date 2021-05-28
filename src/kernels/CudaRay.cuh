#pragma once

#include "math/CudaMath.cuh"

namespace Cuda
{
	#define kSpecularPdf  65000.0f;

	#define kRayAlive 1
	
	struct RayBasic
	{	
		vec3			o;
		vec3			d;
	};

	// The compressed ray object that's stored in the 
	struct CompressedRay
	{
		__device__ CompressedRay() : flags(0) {}

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
		unsigned char	flags;
		unsigned char   depth;

		__device__ void SetAlive() { flags |= kRayAlive; }
		__device__ void Kill() { flags &= ~kRayAlive; }
		__device__ bool IsAlive() { return flags & kRayAlive; }
	};

	// The "full fat" ray objects that most methods will refer to
	struct Ray
	{
		RayBasic od;                // Origin/direction
		float    tNear;             // The parameterised intersection along the ray, defined in cartesian coordinates as o + d * tNear

		vec3	n;                 // Normal at the intersected surface
		vec2	uv;                // UV parameterisation coordinate at the intersected surface
		bool	backfacing;        // Whether the intersection with a forward- or backward-facing surface
		vec3	weight;            // The weight/throughput of the ray depending on context
		float   pdf;               // The value of a probability density function incident to the intersection, depending on context
		float   kickoff;           // The degree to which extant rays should be displaced from the surface to prevent self-intersection
		float   lambda;            // The wavelength of the ray used by the spectral integrator

		__device__ Ray(const CompressedRay& comp) :
			od(comp.od),
			tNear(-FLT_MAX),
			pdf(comp.pdf),
			lambda(comp.lambda),
			weight(comp.weight)
		{
		}
	};

	
}
