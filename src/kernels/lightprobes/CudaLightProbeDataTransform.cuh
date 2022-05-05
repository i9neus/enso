#pragma once

#include "CudaLightProbeGrid.cuh"
#include "kernels/math/CudaSphericalHarmonics.cuh"

namespace Cuda
{
    // Describes how the probe grid data should be transformed between the format used by Probegen and the format used by
    // other software such as Unity.
    struct LightProbeDataTransform
    {
        struct Direction
        {
            Direction(const LightProbeGridParams& gridParams) :
                probeIdx(gridParams.numProbes),
                coeffIdx(gridParams.coefficientsPerProbe * 3),
                sh(gridParams.coefficientsPerProbe) {}

            // Indirection indices between grids
            std::vector<int> probeIdx;

            // Indirect indices between per-channel SH coefficients
            std::vector<int> coeffIdx;

            // Linear transformation of per-channel SH coefficients
            std::vector<mat2> sh;
        };
        
        LightProbeDataTransform(const LightProbeGridParams& gridParams) :
            forward(gridParams),
            inverse(gridParams) { }

        Direction forward;
        Direction inverse;
    };
    
    __host__ LightProbeDataTransform GenerateLightProbeDataTransform(const LightProbeGridParams& gridParams);
}