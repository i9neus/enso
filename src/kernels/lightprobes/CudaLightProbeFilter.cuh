﻿#pragma once

#include "CudaLightProbeGrid.cuh"
#include "../CudaDeviceObjectRAII.cuh"

namespace Json { class Node; }

namespace Cuda
{
#define kMaxCoefficients 5  

    enum LightProbeKernelFilterType : int
    {
        kKernelFilterNull,
        kKernelFilterBox,
        kKernelFilterGaussian,
        kKernelFilterNLM,
        kKernelFilterNLMConst
    };

    struct LightProbeFilterGridData
    {
        __host__ LightProbeFilterGridData& Initialise(AssetHandle<Host::LightProbeGrid>& hostInputGrid,
                                                      AssetHandle<Host::LightProbeGrid>& hostInputHalfGrid,
                                                      AssetHandle<Host::LightProbeGrid>& hostCrossGrid,
                                                      AssetHandle<Host::LightProbeGrid>& hostCrossHalfGrid,
                                                      AssetHandle<Host::LightProbeGrid>& hostOutputGrid,
                                                      AssetHandle<Host::LightProbeGrid>& hostOutputHalfGrid);
        
        // The data to be filtered
        const Device::LightProbeGrid*   cu_inputGrid = nullptr;
        const Device::LightProbeGrid*   cu_inputHalfGrid = nullptr;

        // The data the kernel will be generated from
        const Device::LightProbeGrid*   cu_crossGrid = nullptr;
        const Device::LightProbeGrid*   cu_crossHalfGrid = nullptr;

        // The output from the filter
        Device::LightProbeGrid*         cu_outputGrid = nullptr;
        Device::LightProbeGrid*         cu_outputHalfGrid = nullptr;

        ivec3                           density;
        int                             numProbes;
        int                             totalCoefficients;
        int                             coefficientsPerProbe;
        int                             shCoeffsPerProbe;
        int                             totalSHCoefficients;
    };

    struct LightProbeFilterNLMParams
    {
        enum __tags : int { kHalf, k2xUnder, k2xOver};
        
        __host__ __device__ LightProbeFilterNLMParams();
        __host__ LightProbeFilterNLMParams(const ::Json::Node& node);

        __host__ void ToJson(::Json::Node& node) const;
        __host__ void FromJson(const ::Json::Node& node, const uint flags);

        float                           alpha;
        float                           K;
        int                             patchRadius;
        int                             varianceFormat;
    };

    __host__ __device__ __forceinline__ float ErfApprox(const float x)
    {
        // Closed-form approxiation of the error function.
        // See 'Uniform Approximations for Transcendental Functions', Winitzki 2003, https://doi.org/10.1007/3-540-44839-X_82

        constexpr float a = 8 * (kPi - 3) / (3 * kPi * (4 - kPi));
        return copysign(1.0f, x) * sqrt(1 - expf(-(x * x) * (4 / kPi + a * x * x) / (1 + a * x * x)));
    }

    __host__ __device__ __forceinline__ float AntiderivGauss(const float x, const float sigma)
    {
        // The antiderivative of the normalised Gaussian
        return 0.5f * (1.0f + ErfApprox(x / (sigma * kRoot2)));
    }

    __host__ __device__ __forceinline__ float Integrate1DGaussian(const float t0, const float t1, const float radius)
    {
        // Integrates a Gaussian function whose standard deviation is preset such that its integral between [-radius, radius] is almost 1
        constexpr float sigma = 0.35f;
        return (AntiderivGauss(t1 / radius, sigma) - AntiderivGauss(t0 / radius, sigma));
    }

    __host__ std::vector<float> GenerateGaussianKernel1D(const int kernelSpan);

    __host__ std::vector<float> GenerateBoxKernel1D(const int kernelSpan);

    __device__ float ComputeNLMWeight(LightProbeFilterGridData& gridData, const LightProbeFilterNLMParams& nlmParams, const ivec3& pos0, const ivec3& posK);

    __device__ float ComputeConstVarianceNLMWeight(LightProbeFilterGridData& gridData, const LightProbeFilterNLMParams& nlmParams, const ivec3& pos0, const ivec3& posK);
}