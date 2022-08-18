#pragma once

#include "../CudaSampler.cuh"

using namespace Cuda;

namespace Cuda
{
    namespace Host { template<typename T> class Vector; }
}

namespace GI2D
{
    struct RayBasic2D
    {
        __device__ RayBasic2D() {}
        __device__ RayBasic2D(const vec2& _o, const vec2& _d) : o(_o), d(_d) {}

        vec2        o;
        vec2        d;
    };

    struct Ray2D : public RayBasic2D
    {
        __device__ Ray2D() {}
        __device__ Ray2D(const vec2& _o, const vec2& _d) :
            RayBasic2D(_o, _d) {}
    };

    struct HitCtx2D
    {
        __device__ HitCtx2D() : kickoff(0.f), tFar(kFltMax) {}

        vec2        p;
        vec2        n;
        float		kickoff;
        float       tFar;
    };

    struct ViewTransform
    {
        __host__ __device__ ViewTransform();
        __host__ __device__ ViewTransform(const mat3& mat, const vec2& tra, const float& rot, const float& sca, const vec2& mv, const float& dp) :
            matrix(mat), trans(tra), rotate(rot), scale(sca), mouseView(mv), dPdXY(dp) {}

        mat3 matrix;
        vec2 trans;
        float rotate;
        float scale;

        vec2 mouseView;
        float dPdXY;
    };
}