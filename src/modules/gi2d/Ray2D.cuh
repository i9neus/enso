#pragma once

#include "core/math/Math.cuh"

namespace Enso
{
    namespace Host { template<typename T> class Vector; }

    enum HitCtx2DFlags : int
    {
        kHit2DIsVolume = 1
    };

    struct HitCtx2D
    {
        __host__ __device__ HitCtx2D() : tFar(kFltMax), flags(0) {}

        __host__ __device__ void PrepareNext()
        {
            // Resets the hit context for the next ray test and increments the depth
            tFar = kFltMax;
            flags = 0;
        }

        vec2        p;
        vec2        n;
        float		kickoff;
        float       tFar;
        uint        tracableIdx;
        uchar       flags;
        uchar       depth;
    };

    struct RayRange2D
    {
        __host__ __device__ RayRange2D() : tNear(0.0f), tFar(kFltMax) {}
        __host__ __device__ RayRange2D(const float& tn, const float& tf) : tNear(tn), tFar(tf) {}

        __host__ __device__ __forceinline__ void ClipFar(const float& t) { tFar = fminf(tFar, t); }
        __host__ __device__ __forceinline__ void ClipNear(const float& t) { tNear = fmaxf(tNear, t); }

        float tNear;
        float tFar;
    };

    struct RayBasic2D
    {
        __host__ __device__ RayBasic2D() {}
        __host__ __device__ RayBasic2D(const vec2& _o, const vec2& _d) : o(_o), d(_d) {}

        __host__ __device__ vec2 PointAt(const float t) const { return o + d * t; }

        vec2        o;
        vec2        d;
    };

    enum Ray2DFlags
    {
        kRay2DDirectLightSample = 1,
        kRay2DDirectBxDFSample = 2,
        kRay2DIndirectSample = 4
    };

    struct Ray2D : public RayBasic2D
    {
        __host__ __device__ Ray2D() : flags(0) {}
        __host__ __device__ Ray2D(const vec2& _o, const vec2& _d) :
            RayBasic2D(_o, _d),
            flags(0) {}

        __host__ __device__ __forceinline__ void DeriveIndirectSample(const HitCtx2D& hit, const vec2& extant, const vec3& childWeight)
        {
            o = hit.p + hit.n * hit.kickoff;
            d = extant;
            throughput *= childWeight;
            flags |= kRay2DIndirectSample;
            //pdf = childPdf;
        }

        __host__ __device__ __forceinline__ void DeriveDirectSample(const HitCtx2D& hit, const vec2& extant, const vec3& L, const uint idx)
        {
            o = hit.p + hit.n * hit.kickoff;
            d = extant;
            throughput *= L;
            flags |= kRay2DDirectLightSample;
            lightIdx = idx;
        }

        __host__ __device__ __forceinline__ bool IsDirectSample() const { return flags & (kRay2DDirectLightSample | kRay2DDirectBxDFSample); }
        __host__ __device__ __forceinline__ bool IsIndirectSample() const { return flags & kRay2DIndirectSample; }

        vec3        throughput;
        uchar       flags;
        union
        {
            //float       pdf;
            uint        lightIdx;
        };
    };
}