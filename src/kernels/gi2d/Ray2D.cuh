#pragma once

#include "../math/CudaMath.cuh"

using namespace Cuda;

namespace Cuda
{
    namespace Host { template<typename T> class Vector; }
}

namespace GI2D
{   
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

    struct Ray2D : public RayBasic2D
    {
        __host__ __device__ Ray2D() {}
        __host__ __device__ Ray2D(const vec2& _o, const vec2& _d) :
            RayBasic2D(_o, _d) {}
    };

    struct HitCtx2D
    {
        __host__ __device__ HitCtx2D(const float tf = kFltMax) : kickoff(0.f), tFar(tf) {}

        vec2        p;
        vec2        n;
        float		kickoff;
        float       tFar;
    };
}