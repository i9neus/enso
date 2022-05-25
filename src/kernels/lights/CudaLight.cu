#include "CudaLight.cuh"
#include "../math/CudaColourUtils.cuh"

#include "generic/JsonUtils.h"

namespace Cuda
{
    __host__ __device__ LightParams::LightParams() :
        intensity(1.0f),
        colourHSV(vec3(0.0f, 0.0f, 1.0f)) {}

    __host__ LightParams::LightParams(const ::Json::Node& node) :
        LightParams()
    {
        FromJson(node, ::Json::kSilent);
    }

    __host__ void LightParams::ToJson(::Json::Node& node) const
    {
        renderObject.ToJson(node);
        transform.ToJson(node);

        colourHSV.ToJson("colour", node);
        intensity.ToJson("intensity", node);
    }

    __host__ uint LightParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        renderObject.FromJson(node, flags);
        transform.FromJson(node, flags);

        // TODO: Lights should be able to scale non-uniformally. Evaluation is more complicated, though.
        transform.MakeScaleUniform();

        colourHSV.FromJson("colour", node, flags);
        intensity.FromJson("intensity", node, flags);

        return kRenderObjectDirtyAll;
    }
    
    __host__ void Host::Light::Synchronise()
    {
        Device::Light::Objects objects;
        objects.lightId = m_lightId;
        SynchroniseObjects(static_cast<Device::Light*>(GetDeviceInstance()), objects);

        Log::Debug("Synchronised light '%s'.\n", GetAssetID());
    }

    __host__ AssetHandle<Host::Tracable> Host::Light::GetTracableHandle()
    {
        return nullptr;
    }
}