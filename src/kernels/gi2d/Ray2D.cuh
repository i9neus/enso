#pragma once

#include "../math/CudaMath.cuh"

using namespace Cuda;

namespace Cuda
{
    namespace Host { template<typename T> class Vector; }
}

namespace GI2D
{   
    enum HitCtx2DFlags : int 
    { 
        kHit2DIsVolume = 1 
    };
    
    struct HitCtx2D
    {
        __host__ __device__ HitCtx2D() : tFar(kFltMax), flags(0) {}

        vec2        p;
        vec2        n;
        float		kickoff;
        float       tFar;
        uint        tracableIdx;
        uchar       flags;
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
        kRay2DLightSample = 1
    };

    struct Ray2D : public RayBasic2D
    {           
        __host__ __device__ Ray2D() : flags(0) {}
        __host__ __device__ Ray2D(const vec2& _o, const vec2& _d) :
            RayBasic2D(_o, _d),
            flags(0){}

        __host__ __device__ __forceinline__ void Derive(const HitCtx2D& hit, const vec2& extant, const vec3& childWeight, const float& childPdf, const uchar childFlags)
        {
            o = hit.p + hit.n * hit.kickoff;
            d = extant;
            throughput *= childWeight;
            flags |= childFlags;
            pdf = childPdf;
        }

        __host__ __device__ __forceinline__ void DeriveLightSample(const HitCtx2D& hit, const vec2& extant, const vec3& L, const uint idx)
        {
            o = hit.p + hit.n * hit.kickoff;
            d = extant;
            throughput *= L;
            flags |= kRay2DLightSample;
            lightIdx = idx;
        }

        __host__ __device__ __forceinline__ bool IsLightSample() const { return flags & kRay2DLightSample; }

        vec3        throughput;
        uchar       flags;
        union
        {
            float       pdf;
            uint        lightIdx;
        };
    };

  
}