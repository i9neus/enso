#include "CudaLightProbeFilter.cuh"

#include "generic/JsonUtils.h"

namespace Cuda
{
    __host__ __device__ LightProbeFilterNLMParams::LightProbeFilterNLMParams() : 
        alpha(1.0f),
        K(1.0f),
        patchRadius(1),
        sigma(0.0f),
        varianceFormat(kHalf)
    {
    }

    __host__ LightProbeFilterNLMParams::LightProbeFilterNLMParams(const ::Json::Node& node)
    {

    }

    __host__ void LightProbeFilterNLMParams::ToJson(::Json::Node& node) const
    {
        node.AddValue("alpha", alpha);
        node.AddValue("k", K);
        node.AddValue("sigma", sigma);
        node.AddValue("patchRadius", patchRadius); 
        node.AddEnumeratedParameter("varianceFormat", std::vector<std::string>({ "half", "2xunder", "2xover" }), varianceFormat);        
    }

    __host__ void LightProbeFilterNLMParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        node.GetValue("alpha", alpha, flags);
        node.GetValue("k", K, flags);
        node.GetValue("sigma", sigma, flags);
        node.GetValue("patchRadius", patchRadius, flags);
        node.GetEnumeratedParameter("varianceFormat", std::vector<std::string>({ "half", "2xunder", "2xover" }), varianceFormat, flags);

        alpha = clamp(alpha, 0.0f, std::numeric_limits<float>::max());
        K = clamp(K, 0.0f, std::numeric_limits<float>::max());
        sigma = clamp(sigma, 0.0f, std::numeric_limits<float>::max());
    }

    __host__ LightProbeFilterGridData& LightProbeFilterGridData::Initialise(AssetHandle<Host::LightProbeGrid>& hostInputGrid,
                                                                            AssetHandle<Host::LightProbeGrid>& hostCrossGrid,
                                                                            AssetHandle<Host::LightProbeGrid>& hostCrossHalfGrid,
                                                                            AssetHandle<Host::LightProbeGrid>& hostOutputGrid)
    {
        Assert(hostInputGrid);
        
        cu_inputGrid = hostInputGrid->GetDeviceInstance();
        cu_crossGrid = (hostCrossGrid) ? hostCrossGrid->GetDeviceInstance() : nullptr;
        cu_crossHalfGrid = (hostCrossHalfGrid) ? hostCrossHalfGrid->GetDeviceInstance() : nullptr;
        cu_outputGrid = (hostOutputGrid) ? hostOutputGrid->GetDeviceInstance() : nullptr;

        // Copy some variables out of the input grid to save time
        const auto& gridParams = hostInputGrid->GetParams();
        density = gridParams.gridDensity;
        numProbes = gridParams.numProbes;
        coefficientsPerProbe = gridParams.coefficientsPerProbe;
        shCoeffsPerProbe = gridParams.shCoefficientsPerProbe;
        totalCoefficients = numProbes * coefficientsPerProbe;
        totalSHCoefficients = numProbes * shCoeffsPerProbe;

        return *this;
    }

    __device__ float ComputeNLMWeight(LightProbeFilterGridData& gridData, const LightProbeFilterNLMParams& nlmParams, const ivec3& pos0, const ivec3& posK)
    {
        int numValidWeights = 0;
        float weights[kMaxCoefficients];
        memset(weights, 0, sizeof(float) * kMaxCoefficients);

        if (!gridData.cu_crossGrid || !gridData.cu_crossHalfGrid) { return 0.0f; }

        // Iterate over the local block surrounding each element and compute the relative distance
        for (int w = -nlmParams.patchRadius; w <= nlmParams.patchRadius; ++w)
        {
            for (int v = -nlmParams.patchRadius; v <= nlmParams.patchRadius; ++v)
            {
                for (int u = -nlmParams.patchRadius; u <= nlmParams.patchRadius; ++u)
                {             
                    // If this probe is outside the grid, move on
                    const int probeIdxM = gridData.cu_crossGrid->IdxAt(pos0 + ivec3(u, v, w));
                    const int probeIdxN = gridData.cu_crossGrid->IdxAt(pos0 + posK + ivec3(u, v, w));
                    if (probeIdxM < 0 || probeIdxN < 0) { continue; }

                    const vec3* probeM = gridData.cu_crossGrid->At(probeIdxM);
                    const vec3* probeN = gridData.cu_crossGrid->At(probeIdxN);

                    // If this probe is invalid, exclude it
                    if (probeM[gridData.coefficientsPerProbe - 1].x < 0.5f || probeN[gridData.coefficientsPerProbe - 1].x < 0.5f) { continue; }

                    const vec3* probeHalfM = gridData.cu_crossHalfGrid->At(probeIdxM);
                    const vec3* probeHalfN = gridData.cu_crossHalfGrid->At(probeIdxN);

                    for (int coeffIdx = 0; coeffIdx < gridData.coefficientsPerProbe - 1; ++coeffIdx)
                    {
                        const vec3& M = probeM[coeffIdx];
                        const vec3& N = probeN[coeffIdx];
                        const vec3& halfM = probeHalfM[coeffIdx];
                        const vec3& halfN = probeHalfN[coeffIdx];

                        vec3 varN, varM;
                        switch (nlmParams.varianceFormat)
                        {
                        case LightProbeFilterNLMParams::kHalf:
                        {
                            // Variance buffer receives N/2 samples and is divided by N
                            varN = sqr((N - halfN) - halfN) * 2.0f;
                            varM = sqr((M - halfM) - halfM) * 2.0f;
                        }
                        break;
                        case LightProbeFilterNLMParams::k2xUnder:
                        {
                            // Variance buffer receives N/2 samples and is divided by N/2
                            varN = sqr((N - (halfN * 0.5f)) - (halfN * 0.5f)) * 2.0f;
                            varM = sqr((M - (halfM * 0.5f)) - (halfM * 0.5f)) * 2.0f;
                        }
                        break;
                        case LightProbeFilterNLMParams::k2xOver:
                        {
                            // Variance buffer receives 2N samples and is divided by 2N
                            varN = sqr((halfN - (N * 0.5f)) - (N * 0.5f)) * 4.0f;
                            varM = sqr((halfM - (M * 0.5f)) - (M * 0.5f)) * 4.0f;
                        }
                        break;
                        default:
                            varN = varM = 0.0f;
                        };

                        // Compute the distance from N to M
                        const vec3 d2 = (sqr(M - N) - nlmParams.alpha * (varM + min(varN, varM))) /
                            (vec3(1e-10f) + sqr(nlmParams.K) * (varN + varM));

                        // Derive the weight from the distance
                        weights[coeffIdx] += expf(-max(0.0f, cwiseMax(d2)));
                    }
                    numValidWeights++;
                }
            }
        }

        // Find the coefficient with the maximum weight and return
        float maxWeight = kFltMax;
        for (int coeffIdx = 0; coeffIdx < gridData.coefficientsPerProbe - 1; ++coeffIdx)
        {
            maxWeight = min(maxWeight, weights[coeffIdx]);
        }
        return maxWeight / float(max(1, numValidWeights));
    }

    __device__ float ComputeConstVarianceNLMWeight(LightProbeFilterGridData& gridData, const LightProbeFilterNLMParams& nlmParams, const ivec3& pos0, const ivec3& posK)
    {
        int numValidWeights = 0;
        float weights[kMaxCoefficients];
        memset(weights, 0, sizeof(float) * kMaxCoefficients);

        if (!gridData.cu_crossGrid || !gridData.cu_crossHalfGrid) { return 0.0f; }

        // Iterate over the local block surrounding each element and compute the relative distance
        for (int w = -nlmParams.patchRadius; w <= nlmParams.patchRadius; ++w)
        {
            for (int v = -nlmParams.patchRadius; v <= nlmParams.patchRadius; ++v)
            {
                for (int u = -nlmParams.patchRadius; u <= nlmParams.patchRadius; ++u)
                {
                    // If this probe is outside the grid, move on
                    const int probeIdxM = gridData.cu_crossGrid->IdxAt(pos0 + ivec3(u, v, w));
                    const int probeIdxN = gridData.cu_crossGrid->IdxAt(pos0 + posK + ivec3(u, v, w));
                    if (probeIdxM < 0 || probeIdxN < 0) { continue; }

                    const vec3* probeM = gridData.cu_crossGrid->At(probeIdxM);
                    const vec3* probeN = gridData.cu_crossGrid->At(probeIdxN);

                    // If this probe is invalid, exclude it
                    if (probeM[gridData.coefficientsPerProbe - 1].x < 0.5f || probeN[gridData.coefficientsPerProbe - 1].x < 0.5f) { continue; }

                    for (int coeffIdx = 0; coeffIdx < gridData.coefficientsPerProbe - 1; ++coeffIdx)
                    {
                        const vec3& M = probeM[coeffIdx];
                        const vec3& N = probeN[coeffIdx];       

                        const vec3 d2 = (sqr(M - N) - nlmParams.alpha * nlmParams.sigma) /
                            (vec3(1e-10f) + sqr(nlmParams.K) * nlmParams.sigma);
                        weights[coeffIdx] += expf(-max(0.0f, cwiseMax(d2)));
                    }
                    numValidWeights++;
                }
            }
        }

        // Find the coefficient with the maximum weight and return
        float maxWeight = kFltMax;
        for (int coeffIdx = 0; coeffIdx < gridData.coefficientsPerProbe - 1; ++coeffIdx)
        {
            maxWeight = min(maxWeight, weights[coeffIdx]);
        }
        return maxWeight / float(max(1, numValidWeights));
    }

    __host__ std::vector<float> GenerateGaussianKernel1D(const int kernelSpan)
    {
        // Generates a normalised Gaussian kernel of arbitrary size. The normal distribution is unbounded so we choose a value
        // of sigma that, when integrated, captures >99% of the area under the curve in the range x = [-1, 1].

        Assert(kernelSpan % 2 == 1);

        std::vector<float> kernel;
        kernel.resize(kernelSpan);

        const int kernelRadius = kernelSpan / 2;
        constexpr float sigma = 0.35f;
        constexpr float recipIntegralNorm = 1 / 0.995841f;

        for (int i = 0; i <= kernelRadius; ++i)
        {
            // Compute the sub-interval [x0, x1] that corresponds to the bucket of the kernel element
            const float x0 = (i / float(kernelSpan)) * 2.0 - 1.0f;
            const float x1 = ((i + 1) / float(kernelSpan)) * 2.0 - 1.0f;

            // Integrate the Gaussian
            kernel[i] = kernel[kernelSpan - i - 1] = (AntiderivGauss(x1, sigma) - AntiderivGauss(x0, sigma)) * recipIntegralNorm;
        }

        return kernel;
    }

    __host__ std::vector<float> GenerateBoxKernel1D(const int kernelSpan)
    {
        Assert(kernelSpan % 2 == 1);

        return std::vector<float>(kernelSpan, 1.0f / kernelSpan);
    }
}