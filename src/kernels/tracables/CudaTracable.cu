#include "CudaTracable.cuh"
#include "../materials/CudaMaterial.cuh"
#include "generic/JsonUtils.h"

namespace Cuda
{
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
        
        AssetHandle<Host::Material> materialAsset = GetAssetHandleForBinding<Host::Tracable, Host::Material>(objectContainer, m_materialId);

        // Push the binding to the device
        if (materialAsset)
        {
            SynchroniseObjects(static_cast<Device::Tracable*>(GetDeviceInstance()), materialAsset->GetDeviceInstance());
        }
    }
}