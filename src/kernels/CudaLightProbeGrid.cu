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
        debugOutputPRef = false; 
        debugOutputValidity = false;
        debugBakePRef = false;
        axisSwizzle = kXYZ;
        axisMultiplier = 1.0f;
        invertX = invertY = invertZ = false;
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
        node.AddValue("debugOutputPRef", debugOutputPRef);
        node.AddValue("debugOutputValidity", debugOutputValidity);
        node.AddValue("debugBakePRef", debugBakePRef);

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
        node.GetValue("debugOutputPRef", debugOutputPRef, flags);
        node.GetValue("debugOutputValidity", debugOutputValidity, flags);
        node.GetValue("debugBakePRef", debugBakePRef, flags);

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

    __device__ void Device::LightProbeGrid::SetSHCoefficient(const int probeIdx, const int coeffIdx, const vec3& L)
    {
        CudaDeviceAssert(cu_data);
        CudaDeviceAssert(probeIdx < m_params.numProbes);
        CudaDeviceAssert(coeffIdx < m_params.coefficientsPerProbe);

        const int idx = probeIdx * m_params.coefficientsPerProbe + coeffIdx;
        CudaDeviceAssert(idx < cu_data->Size());

        (*cu_data)[idx] = L;
    }

    __device__ vec3 Device::LightProbeGrid::GetSHCoefficient(const int probeIdx, const int coeffIdx) const
    {
        return (*cu_data)[probeIdx * m_params.coefficientsPerProbe + coeffIdx];
    }

    __device__ inline HitPoint HitToObjectSpace(const HitPoint& world, const BidirectionalTransform& bdt)
    {
        HitPoint object;
        object.p = world.p - bdt.trans;
        object.n = world.n + object.p;
        object.p = bdt.fwd * object.p;
        object.n = normalize((bdt.fwd * object.n) - object.p);
        return object;
    }

    __device__ vec3 Device::LightProbeGrid::InterpolateCoefficient(const ivec3 gridIdx, const uint coeffIdx, const vec3& delta) const
    {
        vec3 vert[8];
        for (int z = 0, idx = 0; z < 2; z++)
        {
            for (int y = 0; y < 2; y++)
            {
                for (int x = 0; x < 2; x++, idx++)
                {
                    const ivec3 vertCoord = gridIdx + ivec3(x, y, z);
                    const int sampleIdx = coeffIdx + m_params.coefficientsPerProbe *
                        (vertCoord.z * (m_params.gridDensity.x * m_params.gridDensity.y) + vertCoord.y * m_params.gridDensity.x + vertCoord.x);

                    assert(sampleIdx < cu_data->Size());

                    vert[idx] = (*cu_data)[sampleIdx];
                }
            }
        }

        // Trilinear interpolate
        return mix(mix(mix(vert[0], vert[1], delta.x), mix(vert[2], vert[3], delta.x), delta.y),
                mix(mix(vert[4], vert[5], delta.x), mix(vert[6], vert[7], delta.x), delta.y), delta.z);
    }

    __device__ vec3 Device::LightProbeGrid::Evaluate(const HitCtx& hitCtx) const
    {  
        CudaDeviceAssert(cu_data);
        
        vec3 pGrid = PointToObjectSpace(hitCtx.hit.p, m_params.transform);

        // Debug the grid 
        if (m_params.debugOutputPRef)
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
        if (m_params.debugOutputValidity)
        {
            return mix(kRed, kGreen, InterpolateCoefficient(gridIdx, m_params.coefficientsPerProbe - 1, delta).x);
        }
        else
        {
            for (int coeffIdx = 0; coeffIdx < m_params.coefficientsPerProbe - 1; coeffIdx++)
            {
                L += InterpolateCoefficient(gridIdx, coeffIdx, delta) * SH::Project(n, coeffIdx);
            }
        }

        return L;
    }

    __host__ Host::LightProbeGrid::LightProbeGrid(const std::string& id)
    {
        MakeChildObject();
        
        m_data = AssetHandle<Host::Array<vec3>>(tfm::format("%s_probeGridData", id), m_hostStream);

        cu_deviceData = InstantiateOnDevice<Device::LightProbeGrid>();

        Cuda::Synchronise(cu_deviceData, m_data->GetDeviceInstance());
    }

    __host__ Host::LightProbeGrid::~LightProbeGrid()
    {
        OnDestroyAsset();
    }

    __host__ void Host::LightProbeGrid::OnDestroyAsset()
    {
        m_data.DestroyAsset();
        DestroyOnDevice(cu_deviceData);
    }
    
    __host__ void Host::LightProbeGrid::Prepare(const LightProbeGridParams& params)
    {
        m_params = params;
        m_params.coefficientsPerProbe = SH::GetNumCoefficients(m_params.shOrder) + 1;
        m_params.numProbes = Volume(m_params.gridDensity);

        const int newSize = m_params.numProbes * m_params.coefficientsPerProbe;
        int arraySize = m_data->Size();

        // Resize the array as a power of two 
        if (arraySize < newSize)
        {
            for (arraySize = 1; arraySize < newSize; arraySize <<= 1) {}

            Log::Debug("Resized to %i\n", arraySize);
            m_data->Resize(arraySize);
        }        

        Cuda::SynchroniseObjects(cu_deviceData, m_params);
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
        m_data->Download(rawData);
    }

    __host__ bool Host::LightProbeGrid::IsValid() const
    {
        return m_data->Size() > 0;
    }
}