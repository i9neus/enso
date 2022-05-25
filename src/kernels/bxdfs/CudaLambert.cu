#include "CudaLambert.cuh"
#include "../lightprobes/CudaLightProbeGrid.cuh"
#include "../cameras/CudaLightProbeCamera.cuh"

#include "generic/JsonUtils.h"

namespace Cuda
{
    __host__ LambertBRDFParams::LambertBRDFParams(const ::Json::Node& node) : LambertBRDFParams() { FromJson(node, ::Json::kSilent); }

    __host__ void LambertBRDFParams::ToJson(::Json::Node& node) const
    {
        node.AddValue("probeGridFlags", probeGridFlags);
        node.AddValue("maxSHOrder", maxSHOrder);
    }

    __host__ uint LambertBRDFParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        node.GetValue("probeGridFlags", probeGridFlags, flags);
        node.GetValue("maxSHOrder", maxSHOrder, flags);

        clamp(maxSHOrder, 0, 2);

        return kRenderObjectClean;
    }
    
    __device__ bool Device::LambertBRDF::Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, const vec2& xi, vec3& extant, float& pdf) const
    {
        // Sample the Lambertian direction
        vec3 r = vec3(SampleUnitDisc(xi), 0.0f);
        r.z = sqrt(1.0 - sqr(r.x) - sqr(r.y));

        pdf = r.z / kPi;
        extant = CreateBasis(hitCtx.hit.n) * r;

        return true;
    }
    __device__ bool Device::LambertBRDF::Evaluate(const vec3& incident, const vec3& extant, const HitCtx& hitCtx, float& weight, float& pdf) const
    {
        weight = dot(extant, hitCtx.hit.n) / kPi;
        pdf = weight;

        return true;
    }

    __device__ vec3 Device::LambertBRDF::EvaluateCachedRadiance(const HitCtx& hitCtx) const
    {
        if (!(m_params.probeGridFlags & kLambertUseProbeGrid)) { return kZero; }

        vec3 L(0.0f);
        if (m_params.probeGridFlags & kLambertGridChannel0 && m_objects.lightProbeGrids[0]) { L += m_objects.lightProbeGrids[0]->Evaluate(hitCtx, m_params.maxSHOrder); }
        if (m_params.probeGridFlags & kLambertGridChannel1 && m_objects.lightProbeGrids[1]) { L += m_objects.lightProbeGrids[1]->Evaluate(hitCtx, m_params.maxSHOrder); }
        if (m_params.probeGridFlags & kLambertGridChannel2 && m_objects.lightProbeGrids[2]) { L += m_objects.lightProbeGrids[2]->Evaluate(hitCtx, m_params.maxSHOrder); }
        if (m_params.probeGridFlags & kLambertGridChannel3 && m_objects.lightProbeGrids[3]) { L += m_objects.lightProbeGrids[3]->Evaluate(hitCtx, m_params.maxSHOrder); }
        return L;
    }

    __host__ AssetHandle<Host::RenderObject> Host::LambertBRDF::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kBxDF) { return AssetHandle<Host::RenderObject>(); }

        return CreateAsset<Host::LambertBRDF>(id, json);
    }

    __host__ Host::LambertBRDF::LambertBRDF(const std::string& id, const ::Json::Node& parentNode) :
        Host::BxDF(id, parentNode),
        cu_deviceData(nullptr)
    {
        cu_deviceData = InstantiateOnDevice<Device::LambertBRDF>(id);

        FromJson(parentNode, ::Json::kSilent);
    }

    __host__ void Host::LambertBRDF::OnDestroyAsset()
    {
        m_hostLightProbeGrids[0] = nullptr;
        m_hostLightProbeGrids[1] = nullptr;
        DestroyOnDevice(GetAssetID(), cu_deviceData);
    }

    __host__ uint Host::LambertBRDF::FromJson(const ::Json::Node& parentNode, const uint flags)
    {      
        uint dirtyFlags = Host::BxDF::FromJson(parentNode, flags);
        dirtyFlags |= m_params.FromJson(parentNode, flags);

        // TODO: This is to maintain backwards compatibility. Deprecate it when no longer required.
        //m_gridIDs[0] = "grid_noisy_direct";
        //m_gridIDs[1] = "grid_noisy_indirect";

        parentNode.GetValue("gridChannel0ID", m_gridIDs[0], flags);
        parentNode.GetValue("gridChannel1ID", m_gridIDs[1], flags);
        parentNode.GetValue("gridChannel2ID", m_gridIDs[2], flags);
        parentNode.GetValue("gridChannel3ID", m_gridIDs[3], flags);

        return dirtyFlags | kRenderObjectDirtyRender;
    }

    __host__ void Host::LambertBRDF::Bind(RenderObjectContainer& sceneObjects)
    {
        m_hostData.m_objects = Device::LambertBRDF::Objects();
       
        for (int gridIdx = 0; gridIdx < kLambertGridNumChannels; ++gridIdx)
        {
            if (!m_gridIDs[gridIdx].empty())
            {
                auto& grid = m_hostLightProbeGrids[gridIdx];
                grid = sceneObjects.FindByID<Host::LightProbeGrid>(m_gridIDs[gridIdx]);
                if (grid)
                {
                    m_hostData.m_objects.lightProbeGrids[gridIdx] = grid->GetDeviceInstance();
                    Log::Write("Bound probe grid %i from camera '%s' to Lambert BRDF '%s'.\n", gridIdx, m_gridIDs[gridIdx], GetAssetID());
                }
                else
                {
                    Log::Error("Error: could not bind probe grid '%s' to Lambert BRDF '%s': grid not found.\n", m_gridIDs[gridIdx], GetAssetID());
                }
            }
        }
        
        Cuda::SynchroniseObjects(cu_deviceData, m_params);
        Cuda::SynchroniseObjects(cu_deviceData, m_hostData.m_objects);
    }

    __host__ void Host::LambertBRDF::OnUpdateSceneGraph(RenderObjectContainer& sceneObjects, const uint dirtyFlags)
    {
        // Do a complete re-bind when the scene graph updates
        Bind(sceneObjects);
    }
}