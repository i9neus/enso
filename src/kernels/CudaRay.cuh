#pragma once

#include "CudaCommonIncludes.cuh"

namespace Cuda
{
	struct RayBasic
	{	
		float3			o;
		float3			d;
	};

	// The "full fat" ray objects that most methods will refer to
	struct Ray
	{
		RayBasic od;                // Origin/direction
		float    tNear;             // The parameterised intersection along the ray, defined in cartesian coordinates as o + d * tNear

		float3   n;                 // Normal at the intersected surface
		float2   uv;                // UV parameterisation coordinate at the intersected surface
		bool     backfacing;        // Whether the intersection with a forward- or backward-facing surface
		float3   weight;            // The weight/throughput of the ray depending on context
		float    pdf;               // The value of a probability density function incident to the intersection, depending on context
		float    kickoff;           // The degree to which extant rays should be displaced from the surface to prevent self-intersection
		float    lambda;            // The wavelength of the ray used by the spectral integrator
	};

	struct PackedRay
	{
		RayBasic		ray;
		float3			weight;
		unsigned int    viewportXy;
		unsigned char	flags;
	};
}
