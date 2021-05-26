#pragma once

#include "math/CudaMath.cuh"

namespace Cuda
{
	struct RayBasic
	{	
		vec3			o;
		vec3			d;
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
	};

	struct PackedRay
	{
		RayBasic		ray;
		float3			weight;
		unsigned int    viewportXy;
		unsigned char	flags;
	};
}
