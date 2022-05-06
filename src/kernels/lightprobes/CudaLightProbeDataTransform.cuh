#pragma once

#include "CudaLightProbeGrid.cuh"
#include "kernels/math/CudaSphericalHarmonics.cuh"

namespace Cuda
{
    // Describes how the probe grid data should be transformed between the format used by Probegen and the format used by
    // other software such as Unity.
    class LightProbeDataTransform
    {
    private:
        struct DirectionalTransform
        {
            __host__ DirectionalTransform() = default;
            __host__ DirectionalTransform(const LightProbeGridParams& gridParams)
            {
                Initialise(gridParams);
            }

            __host__ void Initialise(const LightProbeGridParams& gridParams);

            // Indirection indices between grids
            std::vector<int> probeIdx;

            // Indirect indices between per-channel SH coefficients
            std::vector<int> coeffIdx;

            // Linear transformation of per-channel SH coefficients
            std::vector<mat2> sh;
        };

        DirectionalTransform m_forward;
        DirectionalTransform m_inverse;

        LightProbeGridParams m_gridParams;

    public:
        enum Direction : int { kFoward, kInverse };

        __host__ LightProbeDataTransform() = default;
        __host__ LightProbeDataTransform(const LightProbeGridParams& gridParams)
        {
            Construct(gridParams);
        }

        __host__ void Construct(const LightProbeGridParams& gridParams);

        __host__ void Forward(const std::vector<vec3>& inputData, std::vector<vec3>& outputData) const;
        __host__ void Inverse(const std::vector<vec3>& inputData, std::vector<vec3>& outputData) const;

    private:
        __host__ void Transform(const std::vector<vec3>& inputData, const DirectionalTransform& trans, std::vector<vec3>& outputData);

        __host__ void ConstructSHIndices();
        __host__ void ConstructSHTransforms();
        __host__ void ConstructProbePositionIndices();
    }; 
}