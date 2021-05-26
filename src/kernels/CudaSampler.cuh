#pragma once

#include "math/CudaMath.cuh"

namespace Cuda
{
    using PCGState = uvec4;
    
    // Reverse the bits of 32-bit integer
    uint radicalInverse(uint i)
    {
        i = ((i & 0xffffu) << 16u) | (i >> 16u);
        i = ((i & 0x00ff00ffu) << 8u) | ((i & 0xff00ff00u) >> 8u);
        i = ((i & 0x0f0f0f0fu) << 4u) | ((i & 0xf0f0f0f0u) >> 4u);
        i = ((i & 0x33333333u) << 2u) | ((i & 0xccccccccu) >> 2u);
        i = ((i & 0x55555555u) << 1u) | ((i & 0xaaaaaaaau) >> 1u);
        return i;
    }

    // Permuted congruential generator from "Hash Functions for GPU Rendering" (Jarzynski and Olano) http://jcgt.org/published/0009/03/02/paper.pdf
    void pcgAdvance(PCGState& state)
    {
        state = state * 1664525u + 1013904223u;

        state.x += state.y * state.w;
        state.y += state.z * state.x;
        state.z += state.x * state.y;
        state.w += state.y * state.z;

        state ^= state >> 16u;

        state.x += state.y * state.w;
        state.y += state.z * state.x;
        state.z += state.x * state.y;
        state.w += state.y * state.z;
    }

    // Seed the PCG hash function with the current frame multipled by a prime
    PCGState pcgInitialise(int frame)
    {
        return PCGState(frame * 20219, frame * 7243, frame * 12547, frame * 28573);
    }

    // Generates a tuple of canonical random numbers in the range [0, 1]
#define kPCGRandBias 0.999999
    vec4 rand(PCGState& state)
    {
        return kPCGRandBias * vec4(pcgAdvance(state)) / float(0xffffffffu);
    }

    // Generates a tuple of canonical random number and uses them to sample an input texture
    /*vec4 rand(sampler2D sampler)
    {
        return rand();
        return texelFetch(sampler, (gFragCoord + ivec2(pcgAdvance() >> 16)) % 1024, 0) * kPCGRandBias;
    }*/
}