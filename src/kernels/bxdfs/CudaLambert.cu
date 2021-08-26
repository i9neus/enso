#include "CudaLambert.cuh"
#include "../CudaLightProbeGrid.cuh"
#include "../cameras/CudaLightProbeCamera.cuh"

#include "generic/JsonUtils.h"

namespace Cuda
{
    __host__ LambertBRDFParams::LambertBRDFParams(const ::Json::Node& node) : LambertBRDFParams() { FromJson(node, ::Json::kRequiredWarn); }

    __host__ void LambertBRDFParams::ToJson(::Json::Node& node) const
    {
        node.AddValue("lightProbeGridIndex", lightProbeGridIdx);
    }

    __host__ void LambertBRDFParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        node.GetValue("lightProbeGridIndex", lightProbeGridIdx, ::Json::kRequiredWarn);
        lightProbeGridIdx = clamp(lightProbeGridIdx, 0, 1);
    }
    
    __device__ bool Device::LambertBRDF::Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, vec3& extant, float& pdf) const
    {
        const vec2 xi = renderCtx.rng.Rand<0, 1>();

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
        return (cu_lightProbeGrid) ? cu_lightProbeGrid->Evaluate(hitCtx) : vec3(0.0f);
    }

    __host__ AssetHandle<Host::RenderObject> Host::LambertBRDF::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kBxDF) { return AssetHandle<Host::RenderObject>(); }

        return AssetHandle<Host::RenderObject>(new Host::LambertBRDF(json), id);
    }

    __host__ Host::LambertBRDF::LambertBRDF(const ::Json::Node& parentNode) :
        cu_deviceData(nullptr)
    {
        cu_deviceData = InstantiateOnDevice<Device::LambertBRDF>();

        FromJson(parentNode, ::Json::kRequiredWarn);
    }

    __host__ void Host::LambertBRDF::OnDestroyAsset()
    {
        m_hostLightProbeGrid = nullptr;
        DestroyOnDevice(cu_deviceData);
    }

    __host__ void Host::LambertBRDF::FromJson(const ::Json::Node& parentNode, const uint flags)
    {      
        Host::BxDF::FromJson(parentNode, ::Json::kRequiredWarn);
        m_params.FromJson(parentNode, ::Json::kRequiredWarn);

        parentNode.GetValue("lightProbeGrid", m_lightProbeGridID, ::Json::kRequiredWarn);
    }

    __host__ void Host::LambertBRDF::Bind(RenderObjectContainer& sceneObjects)
    {
        if (m_lightProbeGridID.empty()) { return; }
        
        AssetHandle<Host::LightProbeCamera> probeCamera = sceneObjects.FindByID<Host::LightProbeCamera>(m_lightProbeGridID);
        if (!probeCamera)
        {
            Log::Error("Error: could not bind probe grid '%s' to Lambert BRDF '%s': camera not found.\n", m_lightProbeGridID, GetAssetID());
            return;
        }

        Device::LightProbeGrid* cu_grid = nullptr;
        if (probeCamera->GetLightProbeCameraParams().camera.isActive)
        {
            m_hostLightProbeGrid = probeCamera->GetLightProbeGrid(m_params.lightProbeGridIdx);
            cu_grid = m_hostLightProbeGrid->GetDeviceInstance();
            Log::Write("Bound probe grid %i from camera '%s' to Lambert BRDF '%s'.\n", m_params.lightProbeGridIdx, m_lightProbeGridID, GetAssetID());
        }

        Cuda::SynchroniseObjects(cu_deviceData, cu_grid);
    }

    __host__ void Host::LambertBRDF::OnUpdateSceneGraph(RenderObjectContainer& sceneObjects)
    {
        // Do a complete re-bind when the scene graph updates
        Bind(sceneObjects);
    }
}