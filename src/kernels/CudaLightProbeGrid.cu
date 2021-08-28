#define CUDA_DEVICE_ASSERTS

#include "CudaLightProbeGrid.cuh"
#include "generic/JsonUtils.h"
#include "CudaCtx.cuh"
#include "CudaManagedArray.cuh"

#include "math/CudaSphericalHarmonics.cuh"

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

    __host__ __device__ Device::LightProbeGrid::LightProbeGrid()
    {
        
    }

    __device__ void Device::LightProbeGrid::Synchronise(const LightProbeGridParams& params) 
    { 
        m_params = params;
    }

    __device__ void Device::LightProbeGrid::PrepareValidityGrid()
    {
        if (kKernelIdx >= m_params.numProbes) { return; }

        const ivec3 gridIdx(kKernelIdx % m_params.gridDensity.x, 
                            (kKernelIdx / m_params.gridDensity.x) % m_params.gridDensity.y, 
                            kKernelIdx / (m_params.gridDensity.x * m_params.gridDensity.y));

        uchar validity = 0;
        for (int z = 0, idx = 0; z < 2; z++)
        {
            for (int y = 0; y < 2; y++)
            {
                for (int x = 0; x < 2; x++, idx++)
                {
                    const ivec3 vertCoord = gridIdx + ivec3(x, y, z);
                    const int sampleIdx = (m_params.coefficientsPerProbe - 1) + m_params.coefficientsPerProbe *
                        (vertCoord.z * (m_params.gridDensity.x * m_params.gridDensity.y) + vertCoord.y * m_params.gridDensity.x + vertCoord.x);

                    assert(sampleIdx < m_objects.cu_shData->Size());

                    validity |= (uchar((*m_objects.cu_shData)[sampleIdx].x > 0.5f) << idx);
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

    __device__ vec3 Device::LightProbeGrid::GetSHCoefficient(const int probeIdx, const int coeffIdx) const
    {
        return (*m_objects.cu_shData)[probeIdx * m_params.coefficientsPerProbe + coeffIdx];
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

    __device__ vec3 Device::LightProbeGrid::WeightedInterpolateCoefficient(const ivec3 gridIdx, const uint coeffIdx, const vec3& delta, const uchar validity) const
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
                    const ivec3 vertCoord = gridIdx + ivec3(x, y, z);
                    const int sampleIdx = m_params.coefficientsPerProbe *
                        (vertCoord.z * (m_params.gridDensity.x * m_params.gridDensity.y) + vertCoord.y * m_params.gridDensity.x + vertCoord.x);

                    assert(sampleIdx < m_objects.cu_shData->Size());

                    if(validity & (1 << idx))
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
        ivec3 gridIdx;
        vec3 delta;
        for (int dim = 0; dim < 3; dim++)
        {
            if (pGrid[dim] >= float(m_params.gridDensity[dim] - 1))
            {
                gridIdx[dim] = m_params.gridDensity[dim] - 2;
                delta[dim] = 1.0;
            }
            else
            {
                gridIdx[dim] = int(pGrid[dim]);
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
            return mix(kRed, kGreen, WeightedInterpolateCoefficient(gridIdx, m_params.coefficientsPerProbe - 1, delta, 0xff).x);
        }
        break;
        case kProbeGridHarmonicMean:
        {
            return vec3(WeightedInterpolateCoefficient(gridIdx, m_params.coefficientsPerProbe - 1, delta, GetValidity(gridIdx)).y);
        }
        break; 
        default:
        {
            // Sum the SH coefficients
            const uchar validity = GetValidity(gridIdx);
            for (int coeffIdx = 0; coeffIdx < m_params.coefficientsPerProbe - 1; coeffIdx++)
            {
                L += WeightedInterpolateCoefficient(gridIdx, coeffIdx, delta, validity) * SH::Project(n, coeffIdx);
            }
        }
        }

        return L;
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
        m_params = params;
        m_params.coefficientsPerProbe = SH::GetNumCoefficients(m_params.shOrder) + 1;
        m_params.numProbes = Volume(m_params.gridDensity);
        m_params.aspectRatio = vec3(m_params.gridDensity) / cwiseMax(m_params.gridDensity);

        const int newSize = m_params.numProbes * m_params.coefficientsPerProbe;
        int arraySize = m_shData->Size();

        // Resize the array as a power of two 
        if (arraySize < newSize)
        {
            for (arraySize = 1; arraySize < newSize; arraySize <<= 1) {}

            Log::Debug("Resized to %i\n", arraySize);
            m_shData->Resize(arraySize);
            m_validityData->Resize(arraySize);
        }        

        Cuda::SynchroniseObjects(cu_deviceData, m_params);

        KernelPrepareValidityGrid << < (m_params.numProbes + 255) / 256, 256, 0, m_hostStream >> > (cu_deviceData);
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

    __host__ bool Host::LightProbeGrid::IsValid() const
    {
        return m_shData->Size() > 0;
    }
}