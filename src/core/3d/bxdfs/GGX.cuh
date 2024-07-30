#pragma once

#include "../Ray.cuh"
#include "../Basis.cuh"
#include "BxDF.cuh"

namespace Enso
{
    namespace BxDF
    {
        __host__ __device__ __forceinline__ float G1(const vec3& v, const vec3& m, const vec3& n, const float alpha)
        {
            const float cosTheta = dot(v, n); 
            return step(0.0f, dot(v, m) / dot(v, n)) *
                   2.0f / (1.0f + sqrt(1.0 + sqr(alpha) * sqr(sqrt(1.0 - sqr(cosTheta)) / cosTheta)));
        }
        
        __host__ __device__ float SampleGGX(const vec2& xi, const vec3& i, const vec3& n, const float& alpha, vec3& o, float& weight)
        {
            // Sample the microsurface normal with the GGX distribution
            float thetaM = atan(alpha * sqrt(xi.x) / sqrt(1.0f - xi.x));
            float phiM = kTwoPi * xi.y;
            const float sinThetaM = sin(thetaM);
            const vec3 m = CreateBasis(n) * vec3(cos(phiM) * sinThetaM, sin(phiM) * sinThetaM, cos(thetaM));

            weight = fminf(2.0f, fabsf(dot(i, m)) * G1(i, m, n, alpha) * G1(o, m, n, alpha) / (fabsf(dot(i, n)) * fabsf(dot(m, n))));
            o = Reflect(i, m);
            return 1.;
        }

        __host__ __device__ void EvalauateGGX()
        {

        }
    }
}