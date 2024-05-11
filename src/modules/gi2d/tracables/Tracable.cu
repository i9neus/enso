#include "Tracable.cuh"
#include "core/math/ColourUtils.cuh"
#include "io/json/JsonUtils.h"

namespace Enso
{
    __host__ __device__ bool Device::Tracable::IntersectBBox(const BBox2f& bBox) const
    {
        return bBox.Intersects(SceneObject::m_params.objectBBox);
    }

    __host__ Host::Tracable::Tracable(const Asset::InitCtx& initCtx, Device::Tracable* hostInstance, const AssetHandle<const Host::SceneContainer>& scene) :
        SceneObject(initCtx, hostInstance, scene),
        m_hostInstance(hostInstance)
    {
    }

    __host__ void Host::Tracable::SetDeviceInstance(Device::Tracable* deviceInstance)
    {
        SceneObject::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::SceneObject>(deviceInstance));
        cu_deviceInstance = deviceInstance;
    }

    __host__ void Host::Tracable::OnSynchroniseSceneObject(const uint syncFlags)
    {
        if (syncFlags & kSyncParams)
        {
            SynchroniseObjects<Device::Tracable>(cu_deviceInstance, m_hostInstance->m_params);
        }

        OnSynchroniseTracable(syncFlags);
    }

    __host__ bool Host::Tracable::Serialise(Json::Node& node, const int flags) const
    {
        Json::Node tracableNode = node.AddChildObject("tracable");
        SceneObject::Serialise(tracableNode, flags);

        return true;
    }

    __host__ bool Host::Tracable::Deserialise(const Json::Node& node, const int flags)
    {
        bool isDirty = false;
        Json::Node tracableNode = node.GetChildObject("tracable", flags);
        if (tracableNode) { isDirty |= SceneObject::Deserialise(tracableNode, flags); }

        return isDirty;
    }
}