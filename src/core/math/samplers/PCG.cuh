#pragma once

#include "../Math.cuh"

namespace Enso
{
    class PCG
    {
#define kPCGRandBias 0.999999f

    private:
        uvec4   m_state;

    public:
        __host__ __device__ PCG() { Initialise(0); }
        __host__ __device__ PCG(const uint& seed) { Initialise(seed); }

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
        __host__ __device__  __forceinline__ vec4 Rand()
        {
            Advance();
            return kPCGRandBias * vec4(m_state) / float(0xffffffffu);
        }       
    };
}