#define CUDA_DEVICE_ASSERTS

#include "CudaLightProbeGrid.cuh"
#include "generic/JsonUtils.h"
#include "../CudaCtx.cuh"
#include "../CudaManagedArray.cuh"

#include "../math/CudaSphericalHarmonics.cuh"

namespace Cuda
{
    __host__ __device__ LightProbeGridParams::LightProbeGridParams()
    {
        gridDensity = ivec3(5, 5, 5);
        shOrder = 1;
        useValidity = false;
        outputMode = kProbeGridIrradiance;
        axisSwizzle = kXYZ;
        axisMultiplier = 1.0f;
        invertX = invertY = invertZ = false;
        aspectRatio = vec3(1.0f);
        maxSamplesPerProbe = 0;
    }

    __host__ LightProbeGridParams::LightProbeGridParams(const ::Json::Node& node) :
        LightProbeGridParams()
    {
        FromJson(node, ::Json::kRequiredWarn);
    }

    __host__ void LightProbeGridParams::ToJson(::Json::Node& node) const
    {
        transform.ToJson(node);

        node.AddArray("gridDensity", std::vector<int>({ gridDensity.x, gridDensity.y, gridDensity.z }));
        node.AddValue("shOrder", shOrder);
        node.AddValue("useValidity", useValidity);
        node.AddEnumeratedParameter("outputMode", std::vector<std::string>({ "irradiance", "validity", "harmonicmean", "pref" }), outputMode);

        node.AddValue("invertX", invertX);
        node.AddValue("invertY", invertY);
        node.AddValue("invertZ", invertZ);
        node.AddEnumeratedParameter("axisSwizzle", std::vector<std::string>({ "xyz", "xzy", "yxz", "yzx", "zxy", "zyx" }), axisSwizzle);
    }

    __host__ void LightProbeGridParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        transform.FromJson(node, flags);

        node.GetVector("gridDensity", gridDensity, flags);
        node.GetValue("shOrder", shOrder, flags);
        node.GetValue("useValidity", useValidity, flags);
        node.GetEnumeratedParameter("outputMode", std::vector<std::string>({ "irradiance", "validity", "harmonicmean", "pref" }), outputMode, flags);

        node.GetValue("invertX", invertX, flags);
        node.GetValue("invertY", invertY, flags);
        node.GetValue("invertZ", invertZ, flags);
        node.GetEnumeratedParameter("axisSwizzle", std::vector<std::string>({ "xyz", "xzy", "yxz", "yzx", "zxy", "zyx" }), axisSwizzle, flags);

        gridDensity = clamp(gridDensity, ivec3(2), ivec3(1000));
        shOrder = clamp(shOrder, 0, 2);
        axisMultiplier = vec3(float(invertX) * 2.0f - 1.0f, float(invertY) * 2.0f - 1.0f, float(invertZ) * 2.0f - 1.0f);
    }

    __host__ __device__  bool LightProbeGridParams::operator!=(const LightProbeGridParams& rhs) const
    {
        return gridDensity != rhs.gridDensity ||
               shOrder != rhs.shOrder;
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

        const ivec3 gridPos0 = GridPosFromProbeIdx(kKernelIdx, m_params.gridDensity);

        uchar validity = 0;
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

        (*m_objects.cu_validityData)[kKernelIdx] = validity;
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

    __device__ vec3* Device::LightProbeGrid::At(const int probeIdx) 
    {
        assert(m_objects.cu_shData);
        assert(probeIdx < m_params.numProbes);
        
        return &(*m_objects.cu_shData)[probeIdx * m_params.coefficientsPerProbe];
    }

    __device__ vec3* Device::LightProbeGrid::At(const ivec3& gridIdx)
    {
        const int probeIdx = gridIdx.z * m_params.gridDensity.x * m_params.gridDensity.y + gridIdx.y * m_params.gridDensity.x + gridIdx.x;
        assert(probeIdx < m_params.numProbes);

        return &(*m_objects.cu_shData)[probeIdx * m_params.coefficientsPerProbe];
    }

    __device__ int Device::LightProbeGrid::IdxAt(const ivec3& gridIdx) const
    {
        if(gridIdx.x < 0 || gridIdx.x >= m_params.gridDensity.x ||
           gridIdx.y < 0 || gridIdx.y >= m_params.gridDensity.y ||
           gridIdx.z < 0 || gridIdx.z >= m_params.gridDensity.z) { return -1; }

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

    __device__ vec3 Device::LightProbeGrid::WeightedInterpolateCoefficient(const ivec3 gridPos, const uint coeffIdx, const vec3& delta, const uchar validity) const
    {
        float w[6] = { 1 - delta.x, delta.x, 1 - delta.y, delta.y, 1 - delta.z, delta.z };

        vec3 L(0.0f);
        float sumWeights = 0.0f;
        for (int z = 0, idx = 0; z < 2; z++)
        {
            for (int y = 0; y < 2; y++)
            {
                for (int x = 0; x < 2; x++, idx++)
                {
                    const ivec3 vertCoord = gridPos + ivec3(x, y, z);
                    const int sampleIdx = m_params.coefficientsPerProbe * ProbeIdxFromGridPos(vertCoord, m_params.gridDensity);

                    assert(sampleIdx < m_objects.cu_shData->Size());

                    if (validity & (1 << idx))
                    {
                        const float weight = w[x] * w[2 + y] * w[4 + z];
                        L += (*m_objects.cu_shData)[sampleIdx + coeffIdx] * weight;
                        sumWeights += weight;
                    }
                }
            }
        }

        return L / (sumWeights + 1e-10f);
    }

    __device__ vec3 Device::LightProbeGrid::NearestNeighbourCoefficient(const ivec3 gridPos, const uint coeffIdx) const
    {
        return (*m_objects.cu_shData)[m_params.coefficientsPerProbe * ProbeIdxFromGridPos(gridPos, m_params.gridDensity) + coeffIdx];
    }

    __device__ __forceinline__ uchar Device::LightProbeGrid::GetValidity(const ivec3& gridIdx) const
    {
        return m_params.useValidity ?
            ((*m_objects.cu_validityData)[gridIdx.z * (m_params.gridDensity.x * m_params.gridDensity.y) + gridIdx.y * m_params.gridDensity.x + gridIdx.x]) :
            0xff;
    }

    __device__ vec3 Device::LightProbeGrid::Evaluate(const HitCtx& hitCtx) const
    {
        assert(m_objects.cu_shData);

        vec3 pGrid = PointToObjectSpace(hitCtx.hit.p, m_params.transform) / m_params.aspectRatio;

        // Debug the grid 
        if (m_params.outputMode == kProbeGridPref)
        {
            return clamp(pGrid + vec3(0.5f), vec3(0.0f), vec3(1.0f));
        }

        pGrid = clamp(pGrid + vec3(0.5f), vec3(0.0f), vec3(1.0f)) * vec3(m_params.gridDensity - ivec3(1.0));

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
        case kProbeGridValidity:
        {
            //return mix(kRed, kGreen, WeightedInterpolateCoefficient(gridPos, m_params.coefficientsPerProbe - 1, delta, 0xff).x); 
            //return mix(kRed, kGreen, NearestNeighbourCoefficient(gridPos, m_params.coefficientsPerProbe - 1).x);
            int set = 0;
            const uchar validity = GetValidity(gridPos);
            for (int bit = 0; bit < 8; ++bit) { set += (validity >> bit) & 1; }
            return mix(kRed, kGreen, float(set) / 8.0f);
        }
        break;
        case kProbeGridHarmonicMean:
        {
            return vec3(WeightedInterpolateCoefficient(gridPos, m_params.coefficientsPerProbe - 1, delta, GetValidity(gridPos)).y);
        }
        break;
        default:
        {
            // Sum the SH coefficients
            const uchar validity = GetValidity(gridPos);
            for (int coeffIdx = 0; coeffIdx < m_params.coefficientsPerProbe - 1; coeffIdx++)
            {
                L += WeightedInterpolateCoefficient(gridPos, coeffIdx, delta, validity) * SH::Project(n, coeffIdx);
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
                coeffRange[i].x = aggregate.minMaxCoeffs[i].x;
                coeffRange[i].y = 1.0f / max(1e-10f, aggregate.minMaxCoeffs[i].y - aggregate.minMaxCoeffs[i].x);
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
                localStats[i].minMaxSamples = vec2(kFltMax, 0.0f);
                for (int j = 0; j < 4; ++j)
                {
                    localStats[i].minMaxCoeffs[j] = vec2(kFltMax, -kFltMax);
                }
                localStats[i].meanValidity = 0.0f;
                localStats[i].meanDistance = 0.0f;
                localStats[i].probeCount = 0;
            }
        }

        __syncthreads();

        const int startIdx = (m_params.numProbes - 1) * kKernelIdx / 256;
        const int endIdx = (m_params.numProbes - 1) * (kKernelIdx + 1) / 256;
        localStats[kKernelIdx].probeCount = 1 + endIdx - startIdx;

        for (int i = startIdx; i <= endIdx; i++)
        {
            // Accumulate coefficient ranges for the histograms
            for (int coeffIdx = 0; coeffIdx < m_params.coefficientsPerProbe - 1 && coeffIdx < 4; ++coeffIdx)
            {
                const auto& coeff = At(i)[coeffIdx];
                auto& minMax = localStats[kKernelIdx].minMaxCoeffs[coeffIdx];
                minMax.x = min(minMax.x, /*SignedLog*/(cwiseMin(coeff)));
                minMax.y = max(minMax.y, /*SignedLog*/(cwiseMax(coeff)));
            }

            // Accumulate probe states
            const auto& coeff = At(i)[m_params.coefficientsPerProbe - 1];
            localStats[kKernelIdx].minMaxSamples = vec2(min(localStats[kKernelIdx].minMaxSamples.x, coeff.z), max(localStats[kKernelIdx].minMaxSamples.y, coeff.z));
            localStats[kKernelIdx].meanValidity += coeff.x;
            localStats[kKernelIdx].meanDistance += coeff.y;
        }

        __syncthreads();

        if (kThreadIdx == 0)
        {
            // Accumulate global stats from shared memory
            result = localStats[0];
            result.meanValidity /= float(max(1, localStats[0].probeCount));
            result.meanDistance /= float(max(1, localStats[0].probeCount));
            for (int i = 1; i < 256; i++)
            {
                result.minMaxSamples = vec2(min(localStats[i].minMaxSamples.x, result.minMaxSamples.x), max(localStats[i].minMaxSamples.y, result.minMaxSamples.y));
                for (int j = 0; j < 4; ++j)
                {
                    result.minMaxCoeffs[j] = vec2(min(localStats[i].minMaxCoeffs[j].x, result.minMaxCoeffs[j].x), max(localStats[i].minMaxCoeffs[j].y, result.minMaxCoeffs[j].y));
                }
                result.meanValidity += localStats[i].meanValidity / float(max(1, localStats[i].probeCount));
                result.meanDistance += localStats[i].meanDistance / float(max(1, localStats[i].probeCount));
            }
            result.meanValidity /= 256.0f;
            result.meanDistance /= 256.0f;
        }
    }

    __host__ Host::LightProbeGrid::LightProbeGrid(const std::string& id)
    {
        RenderObject::SetRenderObjectFlags(kRenderObjectIsChild);

        m_shData = AssetHandle<Host::Array<vec3>>(tfm::format("%s_probeGridSHData", id), m_hostStream);
        m_validityData = AssetHandle<Host::Array<uchar>>(tfm::format("%s_probeGridValidityData", id), m_hostStream);

        cu_deviceData = InstantiateOnDevice<Device::LightProbeGrid>();

        m_deviceObjects.cu_shData = m_shData->GetDeviceInstance();
        m_deviceObjects.cu_validityData = m_validityData->GetDeviceInstance();

        Cuda::SynchroniseObjects(cu_deviceData, m_deviceObjects);
    }

    __host__ Host::LightProbeGrid::~LightProbeGrid()
    {
        OnDestroyAsset();
    }

    __host__ void Host::LightProbeGrid::OnDestroyAsset()
    {
        m_shData.DestroyAsset();
        m_validityData.DestroyAsset();

        DestroyOnDevice(cu_deviceData);
    }

    __global__ void KernelPrepareValidityGrid(Device::LightProbeGrid* cu_grid)
    {
        cu_grid->PrepareValidityGrid();
    }

    __host__ void Host::LightProbeGrid::Prepare(const LightProbeGridParams& params)
    {
        Assert(Volume(m_params.gridDensity) > 0);
        Assert(m_params.shOrder >= 0 && m_params.shOrder < 2);
        
        m_params = params;
        m_params.coefficientsPerProbe = SH::GetNumCoefficients(m_params.shOrder) + 1;
        m_params.numProbes = Volume(m_params.gridDensity);
        m_params.aspectRatio = vec3(m_params.gridDensity) / cwiseMax(m_params.gridDensity);
        //m_params.useValidity = false;        

        const int newSize = m_params.numProbes * m_params.coefficientsPerProbe;
        int arraySize = m_shData->Size();

        // Resize the array as a power of two 
        m_validityData->ExpandToNearestPow2(newSize);
        if(m_shData->ExpandToNearestPow2(newSize))
        {
            Log::Debug("Resized to %i\n", m_shData->Size());
        }        

        Cuda::SynchroniseObjects(cu_deviceData, m_params);

        KernelPrepareValidityGrid << < (m_params.numProbes + 255) / 256, 256, 0, m_hostStream >> > (cu_deviceData);
    }

    __host__ void Host::LightProbeGrid::Replace(const LightProbeGrid& other)
    {
        Prepare(other.GetParams());
        
        m_shData->Replace(*(other.m_shData));
        m_validityData->Replace(*(other.m_validityData));
    }

    __host__ void Host::LightProbeGrid::Swap(LightProbeGrid& other)
    {
        Assert(m_params.gridDensity == other.GetParams().gridDensity &&
            m_params.shOrder == other.GetParams().shOrder);

        m_shData->Swap(*(other.m_shData));
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
        m_shData->Upload(rawData);
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

    __host__ const Device::LightProbeGrid::AggregateStatistics& Host::LightProbeGrid::UpdateAggregateStatistics()
    {
        // Compute aggregate statistics (min/max ranges, counts, etc) for the probe grid
        KernelGetProbeGridAggregateStatistics << <1, 256, 0, m_hostStream >> > (cu_deviceData, m_probeAggregateData.GetDeviceObject());

        // Compute coefficient histograms
        KernelComputeProbeGridHistograms << <1, 256, 0, m_hostStream >> > (cu_deviceData, m_probeAggregateData.GetDeviceObject(), m_coeffHistogram.GetDeviceObject());
        IsOk(cudaStreamSynchronize(m_hostStream));

        m_probeAggregateData.Download();
        m_coeffHistogram.Download();

        KernelPrepareValidityGrid << < (m_params.numProbes + 255) / 256, 256, 0, m_hostStream >> > (cu_deviceData);
        IsOk(cudaStreamSynchronize(m_hostStream));

        return *m_probeAggregateData;
    }

    __host__ const Device::LightProbeGrid::AggregateStatistics& Host::LightProbeGrid::GetAggregateStatistics(const uint** histogram) const
    {
        if (histogram)
        {
            *histogram = &(*m_coeffHistogram);
        }
        return *m_probeAggregateData;
    }
}