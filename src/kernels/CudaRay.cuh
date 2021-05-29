#pragma once

#include "math/CudaMath.cuh"

namespace Cuda
{
	#define kSpecularPdf  65000.0f;

	#define kRayAlive 1

	// The "full fat" ray objects that most methods will refer to
	struct Ray
	{
		struct Basic
		{
			vec3			o;
			vec3			d;
			
			Basic() = default;
			Basic(const Basic& other) = default;
			__device__ Basic(const vec3& o_, const vec3& d_) : o(o_), d(d_) {}

			__device__ Ray::Basic ToObjectSpace(const mat4& matrix) const
			{
				Ray::Basic transformed;
				transformed.d = matrix * (d + o);
				transformed.o = matrix * o;
				transformed.d = transformed.d - transformed.o;
				return transformed;
			}
		};
		
		Basic	od;		
		float   tNear;             // The parameterised intersection along the ray, defined in cartesian coordinates as o + d * tNear		
		vec3	weight;            // The weight/throughput of the ray depending on context
		float   pdf;               // The value of a probability density function incident to the intersection, depending on context
		float   lambda;            // The wavelength of the ray used by the spectral integrator			
	};

	struct CompressedRay
	{
		__device__ CompressedRay() : flags(0) {}

		Ray::Basic		od;
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

	__device__ inline Ray DeriveRay(const CompressedRay& comp)
	{
		Ray newRay;
		newRay.od = comp.od;
		newRay.tNear = -FLT_MAX;
		newRay.pdf = comp.pdf;
		newRay.lambda = comp.lambda;
		newRay.weight = comp.weight;
		return newRay;
	}
}
