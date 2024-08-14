#pragma once

#include "../Math.cuh"
#include <random>

namespace Enso
{
    // Generates a tuples of canonical random numbers in the range [0, 1]
    class MersenneTwister
    {
    private:
        std::mt19937 m_mt;
        std::uniform_real_distribution<float> m_dist;

    public:
        __host__ MersenneTwister(const uint seed = 0u) :
            m_mt(seed),
            m_dist(0.f, 1.0f) {}        

        __host__ __forceinline__ float Rand() { return m_dist(m_mt); }
        __host__ __forceinline__ vec2 Rand2() { return vec2(m_dist(m_mt), m_dist(m_mt)); }
        __host__ __forceinline__ vec3 Rand3() { return vec3(m_dist(m_mt), m_dist(m_mt), m_dist(m_mt)); }
        __host__ __forceinline__ vec4 Rand4() { return vec4(m_dist(m_mt), m_dist(m_mt), m_dist(m_mt), m_dist(m_mt)); }
    };
}