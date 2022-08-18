﻿#pragma once

#include "CudaRay.cuh"
#include "generic/Hash.h"

namespace Cuda
{    
    template<int Idx> __host__ __device__ __forceinline__ float GetHaltonCRPrime() { return 0.0f; }
    template<> __host__ __device__ __forceinline__ float GetHaltonCRPrime<0>() { return 1.0f; }
    template<> __host__ __device__ __forceinline__ float GetHaltonCRPrime<1>() { return 2.0f; }
    template<> __host__ __device__ __forceinline__ float GetHaltonCRPrime<2>() { return 3.0f; }
    template<> __host__ __device__ __forceinline__ float GetHaltonCRPrime<3>() { return 5.0f; }
    
    template<int Base>
    __host__ __device__ __forceinline__ float HaltonBase(uint seed)
    {
        return 0;
    }

    // Samples the radix-2 Halton sequence from seed value, i
    template<>
    __host__ __device__ __forceinline__ float HaltonBase<0>(uint i)
    {
        i = ((i & 0xffffu) << 16u) | (i >> 16u);
        i = ((i & 0x00ff00ffu) << 8u) | ((i & 0xff00ff00u) >> 8u);
        i = ((i & 0x0f0f0f0fu) << 4u) | ((i & 0xf0f0f0f0u) >> 4u);
        i = ((i & 0x33333333u) << 2u) | ((i & 0xccccccccu) >> 2u);
        i = ((i & 0x55555555u) << 1u) | ((i & 0xaaaaaaaau) >> 1u);
        return float(i) / float(0xffffffffu);
    }

    template<>
    __host__ __device__ __forceinline__ float HaltonBase<2>(uint seed)
    {
        uint accum = 0u;
        accum += 1162261467u * (seed % 3u); seed /= 3u;
        accum += 387420489u * (seed % 3u); seed /= 3u;
        accum += 129140163u * (seed % 3u); seed /= 3u;
        accum += 43046721u * (seed % 3u); seed /= 3u;
        accum += 14348907u * (seed % 3u); seed /= 3u;
        accum += 4782969u * (seed % 3u); seed /= 3u;
        accum += 1594323u * (seed % 3u); seed /= 3u;
        accum += 531441u * (seed % 3u); seed /= 3u;
        accum += 177147u * (seed % 3u); seed /= 3u;
        accum += 59049u * (seed % 3u); seed /= 3u;
        accum += 19683u * (seed % 3u); seed /= 3u;
        accum += 6561u * (seed % 3u); seed /= 3u;
        accum += 2187u * (seed % 3u); seed /= 3u;
        accum += 729u * (seed % 3u); seed /= 3u;
        accum += 243u * (seed % 3u); seed /= 3u;
        accum += 81u * (seed % 3u); seed /= 3u;
        accum += 27u * (seed % 3u); seed /= 3u;
        accum += 9u * (seed % 3u); seed /= 3u;
        accum += 3u * (seed % 3u); seed /= 3u;
        return float(accum + seed % 3u) / 3486784400.0f;
    }

    template<>
    __host__ __device__ __forceinline__ float HaltonBase<1>(uint seed)
    {
        uint accum = 0u;
        accum += 244140625u * (seed % 5u); seed /= 5u;
        accum += 48828125u * (seed % 5u); seed /= 5u;
        accum += 9765625u * (seed % 5u); seed /= 5u;
        accum += 1953125u * (seed % 5u); seed /= 5u;
        accum += 390625u * (seed % 5u); seed /= 5u;
        accum += 78125u * (seed % 5u); seed /= 5u;
        accum += 15625u * (seed % 5u); seed /= 5u;
        accum += 3125u * (seed % 5u); seed /= 5u;
        accum += 625u * (seed % 5u); seed /= 5u;
        accum += 125u * (seed % 5u); seed /= 5u;
        accum += 25u * (seed % 5u); seed /= 5u;
        accum += 5u * (seed % 5u); seed /= 5u;
        return float(accum + seed % 5u) / 1220703124.0f;
    }

    template<>
    __host__ __device__ __forceinline__ float HaltonBase<3>(uint seed)
    {
        uint accum = 0u;
        accum += 282475249u * (seed % 7u); seed /= 7u;
        accum += 40353607u * (seed % 7u); seed /= 7u;
        accum += 5764801u * (seed % 7u); seed /= 7u;
        accum += 823543u * (seed % 7u); seed /= 7u;
        accum += 117649u * (seed % 7u); seed /= 7u;
        accum += 16807u * (seed % 7u); seed /= 7u;
        accum += 2401u * (seed % 7u); seed /= 7u;
        accum += 343u * (seed % 7u); seed /= 7u;
        accum += 49u * (seed % 7u); seed /= 7u;
        accum += 7u * (seed % 7u); seed /= 7u;
        return float(accum + seed % 7u) / 1977326742.0f;
    }

    template<>
    __host__ __device__ __forceinline__ float HaltonBase<4>(uint seed)
    {
        uint accum = 0u;
        accum += 214358881u * (seed % 11u); seed /= 11u;
        accum += 19487171u * (seed % 11u); seed /= 11u;
        accum += 1771561u * (seed % 11u); seed /= 11u;
        accum += 161051u * (seed % 11u); seed /= 11u;
        accum += 14641u * (seed % 11u); seed /= 11u;
        accum += 1331u * (seed % 11u); seed /= 11u;
        accum += 121u * (seed % 11u); seed /= 11u;
        accum += 11u * (seed % 11u); seed /= 11u;
        return float(accum + seed % 11u) / 2357947690.0f;
    }

    template<>
    __host__ __device__ __forceinline__ float HaltonBase<5>(uint seed)
    {
        uint accum = 0u;
        accum += 62748517u * (seed % 13u); seed /= 13u;
        accum += 4826809u * (seed % 13u); seed /= 13u;
        accum += 371293u * (seed % 13u); seed /= 13u;
        accum += 28561u * (seed % 13u); seed /= 13u;
        accum += 2197u * (seed % 13u); seed /= 13u;
        accum += 169u * (seed % 13u); seed /= 13u;
        accum += 13u * (seed % 13u); seed /= 13u;
        return float(accum + seed % 13u) / 815730720.0f;
    }

    template<>
    __host__ __device__ __forceinline__ float HaltonBase<6>(uint seed)
    {
        uint accum = 0u;
        accum += 24137569u * (seed % 17u); seed /= 17u;
        accum += 1419857u * (seed % 17u); seed /= 17u;
        accum += 83521u * (seed % 17u); seed /= 17u;
        accum += 4913u * (seed % 17u); seed /= 17u;
        accum += 289u * (seed % 17u); seed /= 17u;
        accum += 17u * (seed % 17u); seed /= 17u;
        return float(accum + seed % 17u) / 410338672.0f;
    }

    template<>
    __host__ __device__ __forceinline__ float HaltonBase<7>(uint seed)
    {
        uint accum = 0u;
        accum += 47045881u * (seed % 19u); seed /= 19u;
        accum += 2476099u * (seed % 19u); seed /= 19u;
        accum += 130321u * (seed % 19u); seed /= 19u;
        accum += 6859u * (seed % 19u); seed /= 19u;
        accum += 361u * (seed % 19u); seed /= 19u;
        accum += 19u * (seed % 19u); seed /= 19u;
        return float(accum + seed % 19u) / 893871738.0f;
    }

    template<>
    __host__ __device__ __forceinline__ float HaltonBase<8>(uint seed)
    {
        uint accum = 0u;
        accum += 148035889u * (seed % 23u); seed /= 23u;
        accum += 6436343u * (seed % 23u); seed /= 23u;
        accum += 279841u * (seed % 23u); seed /= 23u;
        accum += 12167u * (seed % 23u); seed /= 23u;
        accum += 529u * (seed % 23u); seed /= 23u;
        accum += 23u * (seed % 23u); seed /= 23u;
        return float(accum + seed % 23u) / 3404825446.0f;
    }

    template<int Idx, int B0>
    __host__ __device__  __forceinline__  void HaltonImpl(const uint seed, float* data)
    {
        data[Idx] = fmodf(HaltonBase<B0>(seed) + GetHaltonCRPrime<B0>() * float(seed) / float(0xffffffffu), 1.0f);
    }

    template<int Idx, int B0, int B1, int... Pack>
    __host__ __device__  __forceinline__  void HaltonImpl(const uint seed, float* data)
    {
        data[Idx] = fmodf(HaltonBase<B0>(seed) + GetHaltonCRPrime<B0>() * float(seed) / float(0xffffffffu), 1.0f);
        HaltonImpl<Idx + 1, B1, Pack...>(seed, data);
    }

    template<typename T, int... Pack>
    __host__ __device__  __forceinline__ T Halton(const uint seed)
    {
        T v;
        HaltonImpl<0, Pack...>(seed, v.data);
        return v;
    } 

    template<int Base>
    __host__ __device__  __forceinline__ float Halton(const uint seed)
    {
        return fmodf(HaltonBase<Base>(seed) + GetHaltonCRPrime<Base>() * float(seed) / float(0xffffffffu), 1.0f);
    }

    class PseudoRNG
    {
        #define kPCGRandBias 0.999999f

    private:
        uvec4   m_state;

    public:
        __host__ __device__ PseudoRNG() {}
        __host__ __device__ PseudoRNG(const uint& seed) { Initialise(seed); }
        __host__ __device__ PseudoRNG(const CompressedRay& ray) 
        { 
            Initialise(HashOf(uint(ray.sampleIdx), 
                              uint(ray.depth) + 9871251u, 
                              ray.accumIdx)); 
        }        

        // Permuted congruential generator from "Hash Functions for GPU Rendering" (Jarzynski and Olano) http://jcgt.org/published/0009/03/02/paper.pdf
        __host__ __device__  __forceinline__ void Advance()
        {
            m_state = m_state * 1664525u + 1013904223u;

            m_state.x += m_state.y * m_state.w;
            m_state.y += m_state.z * m_state.x;
            m_state.z += m_state.x * m_state.y;
            m_state.w += m_state.y * m_state.z;

            m_state ^= m_state >> 16u;

            m_state.x += m_state.y * m_state.w;
            m_state.y += m_state.z * m_state.x;
            m_state.z += m_state.x * m_state.y;
            m_state.w += m_state.y * m_state.z;
        }

        // Seed the PCG hash function with the current frame multipled by a prime
        __host__ __device__  __forceinline__ void Initialise(const uint& seed)
        {
            m_state = uvec4(seed * 20219, seed * 7243, seed * 12547, seed * 28573);
        }

        // Generates a 4-tuple of canonical random numbers in the range [0, 1]
        __host__ __device__  __forceinline__ vec4 Rand4()
        {
            Advance();
            return kPCGRandBias * vec4(m_state) / float(0xffffffffu);
        }

        template<int B0, int B1, int B2, int B3>
        __host__ __device__ __forceinline__ vec4 Rand() { return Rand4(); }
        template<int B0, int B1, int B2>
        __host__ __device__ __forceinline__ vec3 Rand() { return Rand4().xyz; }
        template<int B0, int B1>
        __host__ __device__ __forceinline__ vec2 Rand() { return Rand4().xy; }
        template<int B0>
        __host__ __device__ __forceinline__ float Rand() { return Rand4().x; }
    };

    class QuasiRNG
    {
    public:
        __host__ __device__ QuasiRNG(const uint seed) : haltonSeed(seed) {}
        __host__ __device__ QuasiRNG(const CompressedRay& ray) : 
            haltonSeed(HashOf(uint(ray.sampleIdx),
                        uint(ray.depth) + 9871251u,
                        ray.accumIdx)) {}    

        template<int B0, int B1, int B2, int B3>
        __host__ __device__ __forceinline__ vec4 Rand() const
        {
            return Halton<vec4, B0, B1, B2, B3>(haltonSeed);
        }
        template<int B0, int B1, int B2>
        __host__ __device__ __forceinline__ vec3 Rand() const
        {
            return Halton<vec3, B0, B1, B2>(haltonSeed);
        }
        template<int B0, int B1>
        __host__ __device__ __forceinline__ vec2 Rand() const
        {
            return Halton<vec2, B0, B1>(haltonSeed);
        }
        template<int B0>
        __host__ __device__ __forceinline__ float Rand() const
        {
            return HaltonBase<B0>(haltonSeed);
        }

    private:
        const uint		haltonSeed;
    };

    // Quick and dirty method for sampling the unit disc from two canonical random variables. For a better algorithm, see
    // A Low Distortion Map Between Disk and Square (Shirley and Chiu)
    __host__ __device__ __forceinline__ vec2 SampleUnitDisc(const vec2& xi)
    {
        float phi = xi.y * kTwoPi;
        return vec2(sin(phi), cos(phi)) * sqrt(xi.x);
    }

    __host__ __device__ __forceinline__ vec2 SampleUnitDiscLowDistortion(const vec2& xi)
    {
        float phi, r;
        const float a = 2.0f * xi.x - 1.0f, b = 2.0f * xi.y - 1.0f;

        // From A Low Distortion Map Between Disk and Square (Shirley and Chiu)
        if (a > -b) // region 1 or 2
        {
            if (a > b) // region 1, also |a| > |b|
            {
                r = a;
                phi = (kPi / 4) * (b / a);
            }
            else // region 2, also |b| > |a|
            {
                r = b;
                phi = (kPi / 4) * (2 - (a / b));
            }
        }
        else // region 3 or 4
        {
            if (a < b) // region 3, also |a| >= |b|, a != 0
            {
                r = -a;
                phi = (kPi / 4) * (4 + (b / a));
            }
            else // region 4, |b| >= |a|, but a==0 and b==0 could occur.
            {
                r = -b;
                phi = (b != 0) ? ((kPi / 4) * (6 - (a / b))) : 0;
            }
        }

        return vec2(r * cosf(phi), r * sinf(phi));
    }

    __host__ __device__ __forceinline__ vec3 SampleUnitSphere(vec2 xi)
    {
        xi.x = xi.x * 2.0 - 1.0;
        xi.y *= kTwoPi;

        float sinTheta = sqrt(1.0 - xi.x * xi.x);
        return vec3(cos(xi.y) * sinTheta, xi.x, sin(xi.y) * sinTheta);
    }

    __host__ __device__ __forceinline__ vec3 SampleUnitHemisphere(vec2 xi)
    {
        xi.y *= kTwoPi;

        float sinTheta = sqrt(1.0 - xi.x * xi.x);
        return vec3(cos(xi.y) * sinTheta, sin(xi.y) * sinTheta, xi.x);
    }
}