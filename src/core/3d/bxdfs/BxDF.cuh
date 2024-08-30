#pragma once

#include "../Ray.cuh"

namespace Enso
{
    namespace BxDF
    {
        __host__ __device__ __forceinline__ vec3 Reflect(const vec3& i, const vec3 & n) 
        {
            // As defined in the GLSL spec: https://registry.khronos.org/OpenGL-Refpages/gl4/html/reflect.xhtml
            // Assumes the incident ray to be in the opposite hemisphere to the normal
            return i - 2 * dot(n, i) * n; 
        }

        __host__ __device__ __forceinline__ vec3 Refract(const vec3& i, const vec3& n, const float& eta)
        {
            // Equal to refract() as defined in the GLSL spec: https://registry.khronos.org/OpenGL-Refpages/gl4/html/refract.xhtml
            // Assumes the incident ray to be in the opposite hemisphere to the normal
            const float nDoti = dot(n, i);            
            const float k = 1.f - eta * eta * (1.0f - nDoti*nDoti);
            return (k < 0.0f) ? kZero : (eta * i - (eta * nDoti + sqrt(k)) * n);
        }
        
        __host__ __device__ __forceinline__ float Fresnel(const float cosI, const float eta1, const float eta2)
        {
            const float sinI = sqrt(1.0f - cosI * cosI);
            float beta = 1.0f - sqr(sinI * eta1 / eta2);

            if (beta < 0.0f) { return 1.0f; }

            beta = sqrt(beta);
            return (sqr((eta1 * cosI - eta2 * beta) / (eta1 * cosI + eta2 * beta)) +
                sqr((eta1 * beta - eta2 * cosI) / (eta1 * beta + eta2 * cosI))) * 0.5f;
        }

        // Schlick's approximation of the Fresnel term
        __host__ __device__ __forceinline__ float Schlick(float cosI, float eta1, float eta2)
        {
            float alpha = 1.0f - cosI;
            alpha *= alpha;
            alpha *= alpha;
            return mix(sqr((eta1 - eta2) / (eta1 + eta2)), 1.0f, alpha * (1.0f - cosI));
        }
    }
}