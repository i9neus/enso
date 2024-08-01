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
        kRayDirectSampleLight = 1u << 2,
        kRayDirectSampleBxDF = 1u << 3,
        kRayScattered = 1u << 4,
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
        uchar           depth;

        __host__ __device__ static Ray ConstructDefault()
        {
            Ray ray;
            ray.tNear = kFltMax;
            ray.weight = kOne;
            ray.flags = 0;
            ray.depth = 0;
            return ray;
        }

        __host__ __device__ __forceinline__ void Construct(const vec3& o, const vec3& d, const vec3& kickoff, const vec3& w, const uchar de, const uint& f)
        {
            od.o = o + kickoff;
            od.d = d;
            tNear = kFltMax;
            weight = w;
            flags = f;
            depth = de;
        }

        __host__ __device__ __forceinline__ vec3 Point() const { return od.o + od.d * tNear; }
        __host__ __device__ __forceinline__ vec3 PointAt(const float& t) const { return od.o + od.d * t; }

        __host__ __device__ __forceinline__ void SetFlag(const uint f, const bool set)
        {
            if (set) { flags |= f; }
            else { flags &= ~f; }
        }

        __host__ __device__ __forceinline__ uint InheritedFlags() const { return flags & (kRayCausticPath | kRayScattered); }        
        __host__ __device__ __forceinline__ bool IsDirectSampleLight() const { return flags & kRayDirectSampleLight; }
        __host__ __device__ __forceinline__ bool IsDirectSampleBxDF() const { return flags & kRayDirectSampleBxDF; }
        __host__ __device__ __forceinline__ bool IsDirectSample() const { return flags & (kRayDirectSampleLight | kRayDirectSampleBxDF); }
        __host__ __device__ __forceinline__ bool IsBackfacing() const { return flags & kRayBackfacing; }
        __host__ __device__ __forceinline__ bool IsScattered() const { return flags & kRayScattered; }
    };
}