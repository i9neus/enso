#pragma once

#include "core/math/Math.cuh"

namespace Enso
{
    // Origin/direction 
    struct RayBasic
    {
        vec3            o;
        vec3            d;

        __host__ __device__ __forceinline__ vec3 PointAt(const float& t) const { return o + d * t; }

    };

    enum RayFlags : uint
    {
        kRayBackfacing = 1u,
        kRaySubsurface = 1u << 1,
        kRayDirectSample = 1u << 2,
        kRayScattered = 1u << 3,
        kRayLightSample = 1u << 4,
        kRayProbePath = 1u << 5,
        kRayCausticPath = 1u << 6,
        kRayVolumetricPath = 1u << 7
    };

    // Origin/direction plus attributes
    struct Ray
    {
        RayBasic        od;
        float           tNear;
        vec3            weight;
        uint            flags;

        __host__ __device__ static Ray ConstructDefault()
        {
            Ray ray;
            ray.tNear = kFltMax;
            ray.weight = kOne;
            ray.flags = 0u;
            return ray;
        }

        __host__ __device__ __forceinline__ void Construct(const vec3& o, const vec3& d, const vec3& kickoff, const vec3& w, const uint& f)
        {
            od.o = o + kickoff;
            od.d = d;
            tNear = kFltMax;
            weight = w;
            flags = f;
        }

        __host__ __device__ __forceinline__ vec3 PointAt(const float& t) const { return od.o + od.d * t; }

        __host__ __device__ __forceinline__ void SetFlag(const uint f, const bool set)
        {
            if (set) { flags |= f; }
            else { flags &= ~f; }
        }

        __host__ __device__ __forceinline__ bool IsDirectSample() const { return flags & kRayDirectSample; }
    };
}