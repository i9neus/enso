#define CUDA_DEVICE_ASSERTS

#include "CudaLightProbeGrid.cuh"
#include "generic/JsonUtils.h"
#include "../CudaCtx.cuh"
#include "../CudaManagedArray.cuh"
#include "../math/CudaColourUtils.cuh"

#include "../math/CudaSphericalHarmonics.cuh"

namespace Cuda
{
    static const std::vector<std::string> kSwizzleLabels = { "xyz", "xzy", "yxz", "yzx", "zxy", "zyx" };
    
    __host__ __device__ LightProbeDataTransformParams::LightProbeDataTransformParams() :
        posSwizzle(kXYZ),
        posInvertX(false),
        posInvertY(false),
        posInvertZ(true),
        shSwizzle(kZXY),
        shInvertX(false),
        shInvertY(false),
        shInvertZ(true)
    {
    }

    __host__ LightProbeDataTransformParams::LightProbeDataTransformParams(const ::Json::Node& node) :
        LightProbeDataTransformParams()
    {
        FromJson(node, Json::kSilent);
    }

    __host__ void LightProbeDataTransformParams::ToJson(Json::Node& node) const
    {
        node.AddValue("posInvertX", posInvertX);
        node.AddValue("posInvertY", posInvertY);
        node.AddValue("posInvertZ", posInvertZ);
        node.AddEnumeratedParameter("posSwizzle", kSwizzleLabels, posSwizzle);

        node.AddValue("shInvertX", shInvertX);
        node.AddValue("shInvertY", shInvertY);
        node.AddValue("shInvertZ", shInvertZ);
        node.AddEnumeratedParameter("shSwizzle", kSwizzleLabels, shSwizzle);
    }

    __host__ void LightProbeDataTransformParams::FromJson(const Json::Node& node, const uint flags)
    {
        node.GetValue("posInvertX", posInvertX, flags);
        node.GetValue("posInvertY", posInvertY, flags);
        node.GetValue("posInvertZ", posInvertZ, flags);
        node.GetEnumeratedParameter("posSwizzle", kSwizzleLabels, posSwizzle, flags);

        node.GetValue("shInvertX", shInvertX, flags);
        node.GetValue("shInvertY", shInvertY, flags);
        node.GetValue("shInvertZ", shInvertZ, flags);
        node.GetEnumeratedParameter("shSwizzle", kSwizzleLabels, shSwizzle, flags);
    }

    __host__ __device__ bool LightProbeDataTransformParams::operator!=(const LightProbeDataTransformParams& rhs) const
    {
        return std::memcmp(this, &rhs, sizeof(LightProbeDataTransformParams)) != 0;
    }
    
    __host__ __device__ LightProbeGridParams::LightProbeGridParams()
    {
        gridDensity = ivec3(5, 5, 5);
        clipRegion[0] = ivec3(0, 0, 0);
        clipRegion[1] = gridDensity;
        shOrder = 1;
        outputMode = kProbeGridIrradiance;
        inputColourSpace = kColourSpaceRGB;        
        aspectRatio = vec3(1.0f);
        minMaxSamplesPerProbe = 0;
        dilate = true;
        numProbes = 0;

        Prepare();
    }

    __host__ LightProbeGridParams::LightProbeGridParams(const ::Json::Node& node) :
        LightProbeGridParams()
    {
        FromJson(node, ::Json::kSilent);
    }

    __host__ void LightProbeGridParams::ToJson(::Json::Node& node) const
    {
        transform.ToJson(node);
        dataTransform.ToJson(node);

        node.AddArray("gridDensity", std::vector<int>({ gridDensity.x, gridDensity.y, gridDensity.z }));
        node.AddArray("clipRegionLower", std::vector<int>({ clipRegion[0].x, clipRegion[0].y, clipRegion[0].z }));
        node.AddArray("clipRegionUpper", std::vector<int>({ clipRegion[1].x, clipRegion[1].y, clipRegion[1].z }));
        node.AddValue("shOrder", shOrder);
        node.AddValue("dilate", dilate);
        node.AddEnumeratedParameter("outputMode", std::vector<std::string>({ "irradiance", "validity", "harmonicmean", "pref", "convergence", "sqrerror" }), outputMode);
        node.AddEnumeratedParameter("inputColourSpace", std::vector<std::string>({ "rgb", "xyz", "xyy", "chroma" }), inputColourSpace);       
    }

    __host__ void LightProbeGridParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        transform.FromJson(node, flags);
        dataTransform.FromJson(node, flags);

        node.GetVector("gridDensity", gridDensity, flags);
        gridDensity = clamp(gridDensity, ivec3(2), ivec3(1000));

        clipRegion[0] = ivec3(0, 0, 0);
        clipRegion[1] = gridDensity;
        node.GetVector("clipRegionLower", clipRegion[0], ::Json::kSilent);
        node.GetVector("clipRegionUpper", clipRegion[1], ::Json::kSilent);
        clipRegion[0] = clamp(clipRegion[0], ivec3(0), gridDensity);
        clipRegion[1] = clamp(clipRegion[1], ivec3(0), gridDensity);

        node.GetValue("shOrder", shOrder, flags);
        node.GetValue("dilate", dilate, flags);
        node.GetEnumeratedParameter("outputMode", std::vector<std::string>({ "irradiance", "validity", "harmonicmean", "pref", "convergence", "sqrerror" }), outputMode, flags);
        node.GetEnumeratedParameter("inputColourSpace", std::vector<std::string>({ "rgb", "xyz", "xyy", "chroma" }), inputColourSpace, flags);

        shOrder = clamp(shOrder, 0, 2);
        //axisMultiplier = vec3(float(invertX) * 2.0f - 1.0f, float(invertY) * 2.0f - 1.0f, float(invertZ) * 2.0f - 1.0f);

        Prepare();
    }

    __host__ void LightProbeGridParams::Echo() const
    {
        Log::Indent indent("LightProbeGridParams");
        Log::Debug("Grid density: %s", gridDensity.format());
        Log::Debug("Aspect ratio: %s", aspectRatio.format());
        Log::Debug("SH order: %i", shOrder);
        Log::Debug("Dilation: %s", dilate ? "true" : "false");
        Log::Debug("Coefficients per probe: %i", coefficientsPerProbe);
        Log::Debug("Num probes: %i", numProbes);
        Log::Debug("Min samples: %f", minMaxSamplesPerProbe.x);
        Log::Debug("Max samples: %f", minMaxSamplesPerProbe.y);
    }

    __host__ __device__ void LightProbeGridParams::Prepare()
    {
        shCoefficientsPerProbe = SH::GetNumCoefficients(shOrder);
        coefficientsPerProbe = shCoefficientsPerProbe + 1;
        numProbes = Volume(gridDensity);
    }

    __host__ __device__  bool LightProbeGridParams::operator!=(const LightProbeGridParams& rhs) const
    {
        return gridDensity != rhs.gridDensity ||
               shOrder != rhs.shOrder ||
               inputColourSpace != rhs.inputColourSpace;
    }

    __host__ __device__ Device::LightProbeGrid::LightProbeGrid() { }

    __device__ void Device::LightProbeGrid::Synchronise(const LightProbeGridParams& params)
    {
        m_params = params;
    }

    __device__ void Device::LightProbeGrid::PrepareValidityGrid()
    {
        assert(m_objects.cu_shData);
        if (kKernelIdx >= m_params.numProbes) { return; }

        (*m_objects.cu_validityData)[kKernelIdx] = uchar((*m_objects.cu_shData)[(kKernelIdx + 1) * m_params.coefficientsPerProbe - 1].x > 0.5f);

        /*const ivec3 gridPos0 = GridPosFromProbeIdx(kKernelIdx, m_params.gridDensity);

        /*uchar validity = 0;
        for (int z = 0, idx = 0; z < 2; z++)
        {
            for (int y = 0; y < 2; y++)
            {
                for (int x = 0; x < 2; x++, idx++)
                {
                    if (gridPos0.x == m_params.gridDensity.x - 1 ||
                        gridPos0.y == m_params.gridDensity.y - 1 ||
                        gridPos0.z == m_params.gridDensity.z - 1)
                    {
                        validity |= 1 << idx;
                        continue;
                    }

                    const ivec3 gridPosK = gridPos0 + ivec3(x, y, z);
                    const int gridIdx = m_params.coefficientsPerProbe * (ProbeIdxFromGridPos(gridPosK, m_params.gridDensity) + 1) - 1;

                    assert(gridIdx < m_objects.cu_shData->Size());
                    validity |= uchar((*m_objects.cu_shData)[gridIdx].x > 0.5f) << idx;
                }
            }
        }

        (*m_objects.cu_validityData)[kKernelIdx] = validity;*/
    }

    __device__ void Device::LightProbeGrid::Dilate()
    {
        assert(m_objects.cu_shData);
        assert(m_objects.cu_shData->Size() == m_objects.cu_swapBuffer->Size());
        if (kKernelIdx >= m_params.numProbes) { return; }

        const int gridIdx0 = kKernelIdx * m_params.coefficientsPerProbe;

        // If this probe is valid, just copy the coefficients probe over 
        if ((*m_objects.cu_validityData)[kKernelIdx])
        {
            for (int coeffIdx = 0; coeffIdx < m_params.coefficientsPerProbe; coeffIdx++)
            {
                (*m_objects.cu_swapBuffer)[gridIdx0 + coeffIdx] = (*m_objects.cu_shData)[gridIdx0 + coeffIdx];
            }
            return;
        }

        // Create validity and edge masks to save time later on.
        uint validityMask = 0;
        uint edgeMask = 0;
        const ivec3 gridPos0 = GridPosFromProbeIdx(kKernelIdx, m_params.gridDensity);
        for (int z = -1, idx = 0; z <= 1; z++)
        {
            for (int y = -1; y <= 1; y++)
            {
                for (int x = -1; x <= 1; x++, idx++)
                {
                    if (x == 0 && y == 0 && z == 0) { continue; }

                    const ivec3 gridPosK = gridPos0 + ivec3(x, y, z);
                    if (gridPosK.x < 0 || gridPosK.x >= m_params.gridDensity.x ||
                        gridPosK.y < 0 || gridPosK.y >= m_params.gridDensity.y ||
                        gridPosK.z < 0 || gridPosK.z >= m_params.gridDensity.z)
                    {
                        edgeMask |= 1 << idx;
                        continue;
                    }

                    const uint isValid = (*m_objects.cu_validityData)[ProbeIdxFromGridPos(gridPosK, m_params.gridDensity)];
                    validityMask |= isValid << idx;
                }
            }
        }

        // Clear the swap memory 
        for (int coeffIdx = 0; coeffIdx < m_params.shCoefficientsPerProbe; coeffIdx++)
        {
            (*m_objects.cu_swapBuffer)[gridIdx0 + coeffIdx] = 0.0f;
        }

        // Set the probe metadata to the origin probe. It won't be averaged together like the SH coefficients. 
        (*m_objects.cu_swapBuffer)[gridIdx0 + m_params.shCoefficientsPerProbe] = (*m_objects.cu_shData)[gridIdx0 + m_params.shCoefficientsPerProbe];

        // If the entire neighbourhood is invalid then dilation does nothing. Just set the coefficients to zero and we're done.
        if (validityMask == 0) { return; }

        int sumWeights = 0;
        for (int z = -1, idx = 0; z <= 1; z++)
        {
            for (int y = -1; y <= 1; y++)
            {
                for (int x = -1; x <= 1; x++, idx++)
                {
                    // Neighbour is invalid, so we just skip it
                    if (edgeMask & (1 << idx) || !(validityMask & (1 << idx))) { continue; }

                    const int gridIdxK = m_params.coefficientsPerProbe * ProbeIdxFromGridPos(gridPos0 + ivec3(x, y, z), m_params.gridDensity);

                    // Accumulate coefficients in the swap buffer
                    assert(gridIdxK < m_objects.cu_shData->Size());
                    for (int coeffIdx = 0; coeffIdx < m_params.shCoefficientsPerProbe; coeffIdx++)
                    {
                        (*m_objects.cu_swapBuffer)[gridIdx0 + coeffIdx] += (*m_objects.cu_shData)[gridIdxK + coeffIdx];
                    }
                    sumWeights++;
                }
            }
        }

        // Normalise the coefficients...
        for (int coeffIdx = 0; coeffIdx < m_params.shCoefficientsPerProbe; coeffIdx++)
        {
            (*m_objects.cu_swapBuffer)[gridIdx0 + coeffIdx] /= float(sumWeights);
        }
    }

    __device__ void Device::LightProbeGrid::SetSHCoefficient(const int probeIdx, const int coeffIdx, const vec3& L)
    {
        assert(m_objects.cu_shData);
        assert(probeIdx < m_params.numProbes);
        assert(coeffIdx < m_params.coefficientsPerProbe);

        const int idx = probeIdx * m_params.coefficientsPerProbe + coeffIdx;
        assert(idx < m_objects.cu_shData->Size());

        (*m_objects.cu_shData)[idx] = L;
    }

    __device__ void Device::LightProbeGrid::SetSHLaplacianCoefficient(const int probeIdx, const int coeffIdx, const vec3& L)
    {
        assert(m_objects.cu_shLaplacianData);
        assert(probeIdx < m_params.numProbes);
        assert(coeffIdx < m_params.coefficientsPerProbe);

        const int idx = probeIdx * m_params.coefficientsPerProbe + coeffIdx;
        assert(idx < m_objects.cu_shLaplacianData->Size());

        (*m_objects.cu_shLaplacianData)[idx] = L;
    }

    __device__ vec3* Device::LightProbeGrid::At(const int probeIdx)
    {
        assert(m_objects.cu_shData);
        assert(probeIdx < m_params.numProbes);

        return &(*m_objects.cu_shData)[probeIdx * m_params.coefficientsPerProbe];
    }

    __device__ vec3* Device::LightProbeGrid::LaplacianAt(const int probeIdx)
    {
        assert(m_objects.cu_shData);
        assert(probeIdx < m_params.numProbes);

        return &(*m_objects.cu_shLaplacianData)[probeIdx * m_params.coefficientsPerProbe];
    }

    __device__ vec3* Device::LightProbeGrid::At(const ivec3& gridIdx)
    {
        const int probeIdx = gridIdx.z * m_params.gridDensity.x * m_params.gridDensity.y + gridIdx.y * m_params.gridDensity.x + gridIdx.x;
        assert(probeIdx < m_params.numProbes);

        return &(*m_objects.cu_shData)[probeIdx * m_params.coefficientsPerProbe];
    }

    __device__ int Device::LightProbeGrid::IdxAt(const ivec3& gridIdx) const
    {
        if (gridIdx.x < m_params.clipRegion[0].x || gridIdx.x >= m_params.clipRegion[1].x ||
            gridIdx.y < m_params.clipRegion[0].y || gridIdx.y >= m_params.clipRegion[1].y ||
            gridIdx.z < m_params.clipRegion[0].z || gridIdx.z >= m_params.clipRegion[1].z) {
            return -1;
        }

        return gridIdx.z * m_params.gridDensity.x * m_params.gridDensity.y + gridIdx.y * m_params.gridDensity.x + gridIdx.x;
    }

    __device__ inline HitPoint HitToObjectSpace(const HitPoint& world, const BidirectionalTransform& bdt)
    {
        HitPoint object;
        object.p = world.p - bdt.trans();
        object.n = world.n + object.p;
        object.p = bdt.fwd * object.p;
        object.n = normalize((bdt.fwd * object.n) - object.p);
        return object;
    }

    __device__ vec3 Device::LightProbeGrid::Evaluate(const HitCtx& hitCtx, const int maxSHOrder) const
    {
        assert(m_objects.cu_shData);

        vec3 pGrid = PointToObjectSpace(hitCtx.hit.p, m_params.transform) / m_params.aspectRatio;

        // Return black for out-of-bounds look-ups
        //if (cwiseMin(pGrid) < -0.5f || cwiseMax(pGrid) >= 0.5f) { return kZero; }

        // FIXME: Why do we get a crash if this line is removed?
        pGrid = clamp(pGrid + vec3(0.5f), vec3(0.0f), vec3(1.0f));

        // Debug the grid 
        if (m_params.outputMode == kProbeGridPref) { return pGrid; }

        // Grow normalised coordinate to grid space coordinate
        pGrid = pGrid * vec3(m_params.gridDensity - ivec3(1.0));

        // TODO: Use Cuda's built-in 3D surfaces for more efficient texture indexing

        // Extract the probe xyz indices and deltas
        ivec3 gridPos;
        vec3 delta;
        for (int dim = 0; dim < 3; dim++)
        {
            if (pGrid[dim] >= float(m_params.gridDensity[dim] - 1))
            {
                gridPos[dim] = m_params.gridDensity[dim] - 2;
                delta[dim] = 1.0;
            }
            else
            {
                gridPos[dim] = int(pGrid[dim]);
                delta[dim] = fract(pGrid[dim]);
            }
        }

        // Accumulate each coefficient projected on the normal
        vec3 L(0.0f);
        const vec3& n = hitCtx.hit.n;
        switch (m_params.outputMode)
        {
        case kProbeGridHarmonicMean:
        {
            const float weights = InterpolateCoefficient(*m_objects.cu_shData, gridPos, m_params.coefficientsPerProbe - 1, m_params.coefficientsPerProbe, delta)[kProbeFilterWeights];
            return Hue(0.33f * saturate(1 / weights));
        }
        break;
        case kProbeGridValidity:
        {
            return mix(kRed, kGreen, InterpolateCoefficient(*m_objects.cu_shData, gridPos, m_params.coefficientsPerProbe - 1, m_params.coefficientsPerProbe, delta)[kProbeValidity]);
        }
        break;
        case kProbeGridConvergence:
        {
            if (!m_objects.cu_adaptiveSamplingData) { return kZero; }

            const auto& flags = NearestNeighbourCoefficient(*m_objects.cu_adaptiveSamplingData, gridPos, 0, 1);
            switch (flags)
            {
            case kProbeUnconverged: return kYellow;
            case (kProbeUnconverged | kProbeBelowSampleMin): return kBlue;
            case (kProbeUnconverged | kProbeAtSampleMax): return kRed;
            default: return kGreen;
            }
        }
        break;
        case kProbeGridSqrError:
        {
            if (!m_objects.cu_errorData) { return kZero; }

            float sqrError = InterpolateCoefficient(*m_objects.cu_errorData, gridPos, 0, 1, delta).y;
            if (m_params.camera.samplingMode == kCameraSamplingAdaptiveRelative && m_objects.cu_meanI)
            {
                sqrError /= sqr(*m_objects.cu_meanI * 2.0f);
            }
            const float sqrThreshold = sqr(m_params.camera.errorThreshold);

            constexpr float kHeapmapGain = 1.0f;
            const float impulse = 2.0f / (1.0f + expf(-max(0.0f, sqrError - sqrThreshold))) - 1.0f;
            return Heatmap(saturate(kHeapmapGain * sqrtf(impulse)));
        }
        break;
        default:
        {
            // Sum the SH coefficients
            const int maxSHCoeff = min(m_params.shCoefficientsPerProbe, sqr(maxSHOrder + 1));
            for (int coeffIdx = 0; coeffIdx < maxSHCoeff; coeffIdx++)
            {
                L += InterpolateCoefficient(*m_objects.cu_shData, gridPos, coeffIdx, m_params.coefficientsPerProbe, delta) * SH::Project(n, coeffIdx);
            }

            // Remap to RGB
            switch (m_params.inputColourSpace)
            {
            case kColourSpaceCIEXYZ:
                L = CIEXYZToRGB(L); break;
            case kColourSpaceCIExyY:
                L = CIEXYZToRGB(xyYToXYZ(L)); break;
            case kColourSpaceChroma:
                L = ChromaToRGB(L); break;
            }
        }
        }

        return L;
    }

    __device__ void Device::LightProbeGrid::ComputeProbeGridHistograms(Device::LightProbeGrid::AggregateStatistics& aggregate, uint* coeffHistogram) const
    {
        __shared__ uint sharedHistogram[200];
        __shared__ vec2 coeffRange[4];

        if (kThreadIdx == 0)
        {
            // Initialise the data in the local stats array
            for (int i = 0; i < 200; ++i) { sharedHistogram[i] = 0; }
            for (int i = 0; i < 4; ++i)
            {
                //coeffRange[i].x = -fabs(cwiseExtremum(aggregate.minMaxCoeffs[i]));
                //coeffRange[i].y = 1.0f / (2.0 * -coeffRange[i].x);// max(1e-10f, aggregate.minMaxCoeffs[i].y - aggregate.minMaxCoeffs[i].x);
                coeffRange[i].x = -10.0f;
                coeffRange[i].y = 1.0f / 20.0f;// max(1e-10f, aggregate.minMaxCoeffs[i].y - aggregate.minMaxCoeffs[i].x);
            }
        }

        __syncthreads();

        const int startIdx = (m_params.numProbes - 1) * kKernelIdx / 256;
        const int endIdx = (m_params.numProbes - 1) * (kKernelIdx + 1) / 256;
        for (int i = startIdx; i <= endIdx; i++)
        {
            // Bin the coefficients
            for (int coeffIdx = 0; coeffIdx < m_params.coefficientsPerProbe - 1 && coeffIdx < 4; ++coeffIdx)
            {
                const float extremum = /*SignedLog*/(cwiseExtremum(At(i)[coeffIdx]));
                //const float extremum = /*SignedLog*/(cwiseExtremum(LaplacianAt(i)[coeffIdx]));
                const int binIdx = int(50.0f * clamp((extremum - coeffRange[coeffIdx].x) * coeffRange[coeffIdx].y, 0.0f, nextafterf(1.0f, 0.0f)));

                atomicInc(&sharedHistogram[50 * coeffIdx + binIdx], 0xffffffff);
            }
        }

        __syncthreads();

        if (kThreadIdx == 0)
        {
            memcpy(coeffHistogram, sharedHistogram, sizeof(int) * 200);
        }
    }

    __device__ void Device::LightProbeGrid::GetProbeGridAggregateStatistics(Device::LightProbeGrid::AggregateStatistics& result) const
    {    
        __shared__ AggregateStatistics localStats[256];
        __shared__ Device::LightProbeGrid* grid;

        if (kThreadIdx == 0)
        {
            // Initialise the data in the local stats array
            for (int i = 0; i < 256; i++)
            {
                auto& ls = localStats[i];
                ls.minMaxSamples = vec2(kFltMax, 0.0f);
                ls.meanSamples = 0.0f;
                for (int j = 0; j < Device::LightProbeGrid::AggregateStatistics::kStatsNumCoeffs; ++j)
                {
                    ls.minMaxCoeffs[j] = vec2(kFltMax, -kFltMax);
                    ls.meanSqrIntensity[j] = 0.0f;
                }
                ls.meanValidity = 0.0f;
                ls.meanDistance = 0.0f;
            }
        }

        __syncthreads();

        const int startIdx = (m_params.numProbes - 1) * kKernelIdx / 256;
        const int endIdx = (m_params.numProbes - 1) * (kKernelIdx + 1) / 256;
        auto& ls = localStats[kKernelIdx];

        for (int i = startIdx; i <= endIdx; i++)
        {
            // Accumulate coefficient ranges for the histograms
            for (int coeffIdx = 0; coeffIdx < m_params.coefficientsPerProbe - 1 && coeffIdx < Device::LightProbeGrid::AggregateStatistics::kStatsNumCoeffs; ++coeffIdx)
            {
                const auto& coeff = At(i)[coeffIdx];
                auto& minMax = ls.minMaxCoeffs[coeffIdx];
                minMax.x = min(minMax.x, /*SignedLog*/(cwiseMin(coeff)));
                minMax.y = max(minMax.y, /*SignedLog*/(cwiseMax(coeff)));

                ls.meanSqrIntensity[coeffIdx] += cwiseMax(sqr(coeff));
            }

            // Accumulate probe states
            const auto& coeff = At(i)[m_params.coefficientsPerProbe - 1];
            ls.minMaxSamples = vec2(min(ls.minMaxSamples.x, coeff.z), max(ls.minMaxSamples.y, coeff.z));
            ls.meanSamples += coeff.z;
            ls.meanValidity += coeff.x;
            ls.meanDistance += coeff.y;
        }

        // Normalise the local stats
        const float probeCount = 1 + endIdx - startIdx;
        ls.meanValidity /= probeCount;
        ls.meanDistance /= probeCount;
        ls.meanSamples /= probeCount;
        for (int coeffIdx = 0; coeffIdx < m_params.coefficientsPerProbe - 1 && coeffIdx < Device::LightProbeGrid::AggregateStatistics::kStatsNumCoeffs; ++coeffIdx)
        {            
            ls.meanSqrIntensity[coeffIdx] /= probeCount;
        }

        __syncthreads();

        // Accumulate global stats from shared memory
        if (kThreadIdx == 0)
        {
            result = localStats[0];         
            for (int i = 1; i < 256; i++)
            {
                const auto& ls = localStats[i];
                result.minMaxSamples = vec2(min(ls.minMaxSamples.x, result.minMaxSamples.x), max(ls.minMaxSamples.y, result.minMaxSamples.y));
                result.meanSamples += ls.meanSamples;
                for (int j = 0; j < Device::LightProbeGrid::AggregateStatistics::kStatsNumCoeffs; ++j)
                {
                    result.minMaxCoeffs[j] = vec2(min(ls.minMaxCoeffs[j].x, result.minMaxCoeffs[j].x), max(ls.minMaxCoeffs[j].y, result.minMaxCoeffs[j].y));
                    result.meanSqrIntensity[j] += ls.meanSqrIntensity[j];
                }
                result.meanValidity += ls.meanValidity;
                result.meanDistance += ls.meanDistance;
            }
            for (int j = 0; j < Device::LightProbeGrid::AggregateStatistics::kStatsNumCoeffs; ++j)
            {
                result.meanSqrIntensity[j] /= 256.0f;
            }
            result.meanSamples /= 256.0f;
            result.meanValidity /= 256.0f;
            result.meanDistance /= 256.0f;
        }
    }

    __host__ Host::LightProbeGrid::LightProbeGrid(const std::string& id) :
        RenderObject(id),
        m_hostMeanI(nullptr)
    {
        RenderObject::SetRenderObjectFlags(kRenderObjectIsChild);

        m_shData = CreateChildAsset<Host::Array<vec3>>(tfm::format("%s_probeGridSHData", id), this, m_hostStream);
        m_shLaplacianData = CreateChildAsset<Host::Array<vec3>>(tfm::format("%s_probeGridSHLaplacianData", id), this, m_hostStream);
        m_validityData = CreateChildAsset<Host::Array<uchar>>(tfm::format("%s_probeGridValidityData", id), this, m_hostStream);

        cu_deviceData = InstantiateOnDevice<Device::LightProbeGrid>(id);

        m_deviceObjects.cu_shData = m_shData->GetDeviceInstance();
        m_deviceObjects.cu_shLaplacianData = m_shLaplacianData->GetDeviceInstance();
        m_deviceObjects.cu_validityData = m_validityData->GetDeviceInstance();
        Cuda::SynchroniseObjects(cu_deviceData, m_deviceObjects);

        Prepare();
    }

    __host__ Host::LightProbeGrid::~LightProbeGrid()
    {
        OnDestroyAsset();
    }

    __host__ void Host::LightProbeGrid::OnDestroyAsset()
    {
        m_shData.DestroyAsset();
        m_shLaplacianData.DestroyAsset();
        m_validityData.DestroyAsset();

        DestroyOnDevice(GetAssetID(), cu_deviceData);
    }

    __global__ void KernelPrepareValidityGrid(Device::LightProbeGrid* cu_grid)
    {
        cu_grid->PrepareValidityGrid();
    }

    __global__ void KernelDilate(Device::LightProbeGrid* cu_grid)
    {
        cu_grid->Dilate();
    }   

    __host__ void Host::LightProbeGrid::Prepare(const LightProbeGridParams& params)
    {
        Assert(Volume(m_params.gridDensity) > 0);
        Assert(m_params.shOrder >= 0 && m_params.shOrder < 2);
        
        m_params = params;
        
        Prepare();        
    }

    __host__ void Host::LightProbeGrid::Prepare()
    {
        m_params.coefficientsPerProbe = SH::GetNumCoefficients(m_params.shOrder) + 1;
        m_params.numProbes = Volume(m_params.gridDensity);
        m_params.aspectRatio = vec3(m_params.gridDensity) / cwiseMax(m_params.gridDensity);      

        const int newSize = m_params.numProbes * m_params.coefficientsPerProbe;
        int arraySize = m_shData->Size();

        // Resize the array as a power of two 
        // TODO: Managed arrays should have a notion of size vs. capacity. Add support for this.
        m_validityData->ExpandToNearestPow2(newSize);
        m_shLaplacianData->ExpandToNearestPow2(newSize);
        if (m_shData->ExpandToNearestPow2(newSize))
        {
            Log::Debug("Resized to %i\n", m_shData->Size());
        }

        Cuda::SynchroniseObjects(cu_deviceData, m_params);
    }

    __host__ void Host::LightProbeGrid::Replace(const LightProbeGrid& other)
    {
        // We assume that the other object's parameter structure is fully initialised
        m_params = other.GetParams();
        Cuda::SynchroniseObjects(cu_deviceData, m_params);
        
        m_shData->Replace(*(other.m_shData));
        m_shLaplacianData->Replace(*(other.m_shLaplacianData));
        m_validityData->Replace(*(other.m_validityData));
    }

    __host__ void Host::LightProbeGrid::Swap(LightProbeGrid& other)
    {
        Assert(m_params.gridDensity == other.GetParams().gridDensity &&
               m_params.shOrder == other.GetParams().shOrder);

        m_shData->Swap(*(other.m_shData));
        m_shLaplacianData->Swap(*(other.m_shLaplacianData));
        m_validityData->Swap(*(other.m_validityData));
    }

    __host__ void Host::LightProbeGrid::FromJson(const ::Json::Node& node, const uint flags)
    {
        std::string usdExportPath;
        if (node.GetValue("usdExportPath", usdExportPath, ::Json::kSilent))
        {
            m_usdExportPath = usdExportPath;
            Log::Debug("USD export path: %s\n", usdExportPath);
        }
    }

    __host__ void Host::LightProbeGrid::GetRawData(std::vector<vec3>& rawData) const
    {
        m_shData->Download(rawData);
    }

    __host__ void Host::LightProbeGrid::SetRawData(const std::vector<vec3>& rawData)
    {
        // FIXME: Size of data returned by Download is not the same as that expected by Upload.
        //AssertMsgFmt(rawData.size() == m_params.numProbes * m_params.coefficientsPerProbe,

        AssertMsgFmt(rawData.size() >= m_params.numProbes * m_params.coefficientsPerProbe, 
            "Raw data has size %i; expected %i", rawData.size(), m_params.numProbes * m_params.coefficientsPerProbe);

        m_shData->Upload(rawData);
        
    }

    __host__ void Host::LightProbeGrid::SetOutputMode(const int& mode)
    {
        m_params.outputMode = mode;
        SynchroniseObjects(cu_deviceData, m_params);
    }

    __host__ void Host::LightProbeGrid::SetExternalBuffers(AssetHandle<Host::Array<uchar>> adaptiveSamplingData, AssetHandle<Host::Array<vec2>> errorData, DeviceObjectRAII<float>& meanI)
    {
        m_adaptiveSamplingData = adaptiveSamplingData;
        m_errorData = errorData;
        m_hostMeanI = &meanI;

        m_deviceObjects.cu_adaptiveSamplingData = m_adaptiveSamplingData->GetDeviceInstance();
        m_deviceObjects.cu_errorData = m_errorData->GetDeviceInstance();
        m_deviceObjects.cu_meanI = m_hostMeanI->GetDeviceInstance();

        Cuda::SynchroniseObjects(cu_deviceData, m_deviceObjects);
    }

    __host__ bool Host::LightProbeGrid::IsValid() const
    {
        return m_shData->Size() > 0;
    }

    __global__ void KernelGetProbeGridAggregateStatistics(Device::LightProbeGrid* camera, Device::LightProbeGrid::AggregateStatistics* stats)
    {
        assert(stats);
        camera->GetProbeGridAggregateStatistics(*stats);
    }

    __global__ void KernelComputeProbeGridHistograms(Device::LightProbeGrid* camera, Device::LightProbeGrid::AggregateStatistics* stats, uint* histogram)
    {
        assert(stats);
        assert(histogram);
        camera->ComputeProbeGridHistograms(*stats, histogram);
    }

    __host__ const Host::LightProbeGrid::AggregateStatistics& Host::LightProbeGrid::UpdateAggregateStatistics(const int maxSamples)
    {
        // Compute aggregate statistics (min/max ranges, counts, etc) for the probe grid
        KernelGetProbeGridAggregateStatistics << <1, 256, 0, m_hostStream >> > (cu_deviceData, m_probeAggregateData.GetDeviceInstance());

        // Compute coefficient histograms
        KernelComputeProbeGridHistograms << <1, 256, 0, m_hostStream >> > (cu_deviceData, m_probeAggregateData.GetDeviceInstance(), m_statistics.coeffHistogram.GetDeviceInstance());
        IsOk(cudaStreamSynchronize(m_hostStream));

        m_probeAggregateData.Download();
        m_statistics.coeffHistogram.Download();      

        // Copy the data into the aggregate object
        *static_cast<Device::LightProbeGrid::AggregateStatistics*>(&m_statistics) = *m_probeAggregateData;        
        m_statistics.isConverged = maxSamples > 0 && m_statistics.minMaxSamples.x >= maxSamples;

        return m_statistics;
    }

    __host__ void Host::LightProbeGrid::Integrate()
    {
        if (!m_params.dilate) { return; }
        
        const int gridSize = (m_params.numProbes + 255) / 256;
        KernelPrepareValidityGrid << < gridSize, 256, 0, m_hostStream >> > (cu_deviceData);
        IsOk(cudaStreamSynchronize(m_hostStream));
        
        // Single-pass dilate the probe data to fill in holes and prevent shadow leaks
        KernelDilate<< < gridSize, 256, 0, m_hostStream >> > (cu_deviceData);
        IsOk(cudaStreamSynchronize(m_hostStream));

        // Swap the buffers
        m_shData->Swap(*m_shLaplacianData);
    }

    __host__ void Host::LightProbeGrid::Clear()
    {
        // Reset the SH data
        m_shData->Clear();
    }

    __host__ bool Host::LightProbeGrid::HasSemaphore(const std::string& tag) const
    {
        return m_semaphoreRegistry.find(tag) != m_semaphoreRegistry.end();
    }

    __host__ int Host::LightProbeGrid::GetSemaphore(const std::string& tag) const
    {
        auto it = m_semaphoreRegistry.find(tag);
        return (it != m_semaphoreRegistry.end()) ? it->second : std::numeric_limits<int>::min();
    }

    __host__ void Host::LightProbeGrid::SetSemaphore(const std::string& tag, const int data)
    {
        m_semaphoreRegistry[tag] = data;
    }

    __host__ void Host::LightProbeGrid::PushClipRegion(const ivec3* region)
    {
        m_clipRegionStack.emplace_back(m_params.clipRegion[0], m_params.clipRegion[1]);
        
        m_params.clipRegion[0] = clamp(region[0], ivec3(0), m_params.gridDensity);
        m_params.clipRegion[1] = clamp(region[1], ivec3(0), m_params.gridDensity);

        Cuda::SynchroniseObjects(cu_deviceData, m_params);
    }

    __host__ void Host::LightProbeGrid::PopClipRegion()
    {
        AssertMsg(!m_clipRegionStack.empty(), "Clip region stack is empty.");

        m_params.clipRegion[0] = m_clipRegionStack.back().first;
        m_params.clipRegion[1] = m_clipRegionStack.back().second;
        m_clipRegionStack.pop_back();

        Cuda::SynchroniseObjects(cu_deviceData, m_params);
    }
}