#pragma once

#include "core/math/Math.cuh"

namespace Enso
{
    class OwenSobol
    {
    private:
        uint        m_seed;
        uint        m_sampleIdx;

    public:
        __device__ OwenSobol();

        __device__ void Initialise(const uint seed, const uint sampleIdx);
        __device__ vec4 Rand(const uint dim);
    };
}