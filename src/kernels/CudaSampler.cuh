﻿#pragma once

#include "math/CudaMath.cuh"

namespace Cuda
{    
    // Reverse the bits of 32-bit integer
    __device__ inline uint RadicalInverse(uint i)
    {
        i = ((i & 0xffffu) << 16u) | (i >> 16u);
        i = ((i & 0x00ff00ffu) << 8u) | ((i & 0xff00ff00u) >> 8u);
        i = ((i & 0x0f0f0f0fu) << 4u) | ((i & 0xf0f0f0f0u) >> 4u);
        i = ((i & 0x33333333u) << 2u) | ((i & 0xccccccccu) >> 2u);
        i = ((i & 0x55555555u) << 1u) | ((i & 0xaaaaaaaau) >> 1u);
        return i;
    }

    // Samples the radix-2 Halton sequence from seed value, i
    __device__ inline float HaltonBase2(uint seed)
    {
        return float(RadicalInverse(seed)) / float(0xffffffffu);
    }

    __device__ inline float HaltonBase3(uint seed)
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

    __device__ inline float HaltonBase5(uint seed)
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

    __device__ inline float HaltonBase7(uint seed)
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

    __device__ inline float HaltonBase11(uint seed)
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

    __device__ inline float HaltonBase13(uint seed)
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

    __device__ inline float HaltonBase17(uint seed)
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

    __device__ inline float HaltonBase19(uint seed)
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

    __device__ inline float HaltonBase23(uint seed)
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

    class PCG
    {
    private:
        uvec4   m_state;

    public:
        __device__ PCG(const uint& seed) { Initialise(seed); }

#define kPCGRandBias 0.999999f

        // Permuted congruential generator from "Hash Functions for GPU Rendering" (Jarzynski and Olano) http://jcgt.org/published/0009/03/02/paper.pdf
        __device__  inline void Advance()
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
        __device__  inline void Initialise(const uint& seed)
        {
            m_state = uvec4(seed * 20219, seed * 7243, seed * 12547, seed * 28573);
        }

        // Generates a tuple of canonical random numbers in the range [0, 1]
        __device__  vec4 Rand(const uint& seed)
        {
            Initialise(seed);
            return kPCGRandBias * vec4(m_state) / float(0xffffffffu);
        }

        // Generates a tuple of canonical random numbers in the range [0, 1]
        __device__  vec4 Rand()
        {
            Advance();
            return kPCGRandBias * vec4(m_state) / float(0xffffffffu);
        }

        __device__ inline vec4 operator()() { return Rand(); }
    };

    // Quick and dirty method for sampling the unit disc from two canonical random variables. For a better algorithm, see
    // A Low Distortion Map Between Disk and Square (Shirley and Chiu)
    __device__ inline vec2 SampleUnitDisc(const vec2& xi)
    {
        float phi = xi.y * kTwoPi;
        return vec2(sin(phi), cos(phi)) * sqrt(xi.x);
    }

    __device__ inline vec3 SampleUnitSphere(vec2 xi)
    {
        xi.x = xi.x * 2.0 - 1.0;
        xi.y *= kTwoPi;

        float sinTheta = sqrt(1.0 - xi.x * xi.x);
        return vec3(cos(xi.y) * sinTheta, xi.x, sin(xi.y) * sinTheta);
    }
}