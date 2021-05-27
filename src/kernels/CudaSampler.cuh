#pragma once

#include "math/CudaMath.cuh"

namespace Cuda
{    
    namespace Device
    {
        // Reverse the bits of 32-bit integer
        __device__ inline uint radicalInverse(uint i)
        {
            i = ((i & 0xffffu) << 16u) | (i >> 16u);
            i = ((i & 0x00ff00ffu) << 8u) | ((i & 0xff00ff00u) >> 8u);
            i = ((i & 0x0f0f0f0fu) << 4u) | ((i & 0xf0f0f0f0u) >> 4u);
            i = ((i & 0x33333333u) << 2u) | ((i & 0xccccccccu) >> 2u);
            i = ((i & 0x55555555u) << 1u) | ((i & 0xaaaaaaaau) >> 1u);
            return i;
        }

        // Samples the radix-2 Halton sequence from seed value, i
        __device__ inline float haltonBase2(uint i)
        {
            return float(radicalInverse(i)) / float(0xffffffffu);
        }

        // Quick and dirty method for sampling the unit disc from two canonical random variables. For a better algorithm, see
        // A Low Distortion Map Between Disk and Square (Shirley and Chiu)
        __device__ inline vec2 sampleUnitDisc(const vec2& xi)
        {
            float phi = xi.y * kTwoPi;
            return vec2(sin(phi), cos(phi)) * sqrt(xi.x);
        }

        __device__ inline vec3 sampleUnitSphere(vec2 xi)
        {
            xi.x = xi.x * 2.0 - 1.0;
            xi.y *= kTwoPi;

            float sinTheta = sqrt(1.0 - xi.x * xi.x);
            return vec3(cos(xi.y) * sinTheta, xi.x, sin(xi.y) * sinTheta);
        }

        class PCG
        {
        private:
            uvec4   m_state;

        public:
            #define kPCGRandBias 0.999999f
            
            // Permuted congruential generator from "Hash Functions for GPU Rendering" (Jarzynski and Olano) http://jcgt.org/published/0009/03/02/paper.pdf
            __device__  void Advance()
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
            __device__  void Initialise(int frame)
            {
                m_state = uvec4(frame * 20219, frame * 7243, frame * 12547, frame * 28573);
            }

            // Generates a tuple of canonical random numbers in the range [0, 1]
            __device__  vec4 Rand()
            {
                Advance();
                return kPCGRandBias * vec4(m_state) / float(0xffffffffu);
            }

            __device__ inline vec4 operator()() { return Rand(); }
        };
    }
}