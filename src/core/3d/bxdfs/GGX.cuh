#pragma once

#include "../Ray.cuh"
#include "../Basis.cuh"
#include "BxDF.cuh"

namespace Enso
{
    namespace BxDF
    {
        __device__ __forceinline__ float D_GGX(const vec3& m, const vec3& n, float alpha)
        {
            alpha *= alpha;
            float cosThetaM = clamp(dot(m, n), -1., 1.);
            float tan2ThetaM = 1. / sqr(cosThetaM) - 1.;
            return alpha * step(0.0f, cosThetaM) /
                (kPi * pow4(cosThetaM) * sqr(alpha + tan2ThetaM));
        }

        __device__ __forceinline__ float G1_GGX(vec3 v, vec3 m, vec3 n, float alpha)
        {
            const float cosThetaV = fmaxf(1e-10f, dot(v, n));
            //float tan2ThetaV = sqr(tan(acos(cosThetaV)));
            const float tan2ThetaV = 1.f / sqr(cosThetaV) - 1.f;
            return step(0.0f, dot(v, m) / cosThetaV) * 2.f /
                (1.0f + sqrt(1.0f + sqr(alpha) * tan2ThetaV));
        }

        __device__ __forceinline__ float G_GGX(vec3 i, vec3 o, vec3 m, vec3 n, float alpha)
        {
            return G1_GGX(i, m, n, alpha) * G1_GGX(o, m, n, alpha);
        }

        __device__ __forceinline__ float Weight_GGX(vec3 i, vec3 o, vec3 m, vec3 n, float alpha)
        {
            constexpr float kGGXWeightClamp = 1e3f;
            return fminf(kGGXWeightClamp, fabsf(dot(i, m)) * G_GGX(i, o, m, n, alpha) / (fabsf(dot(i, n) * dot(m, n))));
        }

        __device__ float EvaluateMicrofacetReflectorGGX(const vec3& i, const vec3& o, const vec3& n, const float alpha)
        {
            constexpr float kGGXPDFClamp = 50.f;
            const vec3 hr = normalize(sign(dot(i, n)) * (i + o));

            const float pdf = G_GGX(i, o, hr, n, alpha) * D_GGX(hr, n, alpha) / (4.f * fabsf(dot(i, n) * dot(o, n)));
            return fminf(kGGXPDFClamp, pdf);
        }

        __device__ float SampleMicrofacetReflectorGGX(const vec2& xi, const vec3& i, const vec3& n, const float alpha, vec3& o, float& weight)
        {
            // Sample the microsurface normal with the GGX distribution
            float thetaM = atan(alpha * sqrt(xi.x) / sqrt(1.0f - xi.x));
            float phiM = kTwoPi * xi.y;
            float sinThetaM = sin(thetaM);
            vec3 m = CreateBasis(n) * vec3(cos(phiM) * sinThetaM, sin(phiM) * sinThetaM, cos(thetaM));
            
            o = BxDF::Reflect(-i, m);
            weight = Weight_GGX(i, o, m, n, alpha);

            return (dot(o, n) <= 0.f) ? 0.f : EvaluateMicrofacetReflectorGGX(i, o, m, alpha);
        }
    }
}