#include "CudaTracable.cuh"
#include "../materials/CudaMaterial.cuh"
#include "generic/JsonUtils.h"

namespace Cuda
{

    __host__ __device__ TracableParams::TracableParams()
    {
    }

    __host__ void TracableParams::ToJson(::Json::Node& node) const
    {
        renderObject.ToJson(node);
        transform.ToJson(node);
    }

    __host__ void TracableParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        renderObject.FromJson(node, flags);
        transform.FromJson(node, flags);
    }
    
    __host__ void Host::Tracable::FromJson(const ::Json::Node& node, const uint flags)
    {        
        node.GetValue("material", m_materialId, flags);
    }
    
    __host__ void Host::Tracable::Bind(RenderObjectContainer& objectContainer)
    {
        if (m_materialId.empty())
        {
            Log::Error("Error: no material binding ID was specified for tracable '%s'.\n", GetAssetID());
            return;
        }
                
        // Get a handle to the material asset for this tracable
        m_materialAsset = GetAssetHandleForBinding<Host::Tracable, Host::Material>(objectContainer, m_materialId);        
    }

    __host__ void Host::Tracable::Synchronise()
    {
        Device::Tracable::Objects deviceObjects;
        deviceObjects.lightId = m_lightId;
        if (m_materialAsset)
        {
            deviceObjects.cu_material = m_materialAsset->GetDeviceInstance();
        }

        // Push the binding to the device
        SynchroniseObjects(static_cast<Device::Tracable*>(GetDeviceInstance()), deviceObjects);

        Log::Debug("Synchronised tracable '%s'.\n", GetAssetID());
    }
}