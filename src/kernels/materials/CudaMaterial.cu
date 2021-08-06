#include "CudaMaterial.cuh"
#include "generic/JsonUtils.h" 
#include "../bxdfs/CudaBxDF.cuh"

namespace Cuda
{
    __host__ void Host::Material::FromJson(const ::Json::Node& parentNode, const uint flags)
    {
        Host::RenderObject::UpdateDAGPath(parentNode);
        
        parentNode.GetValue("bxdf", m_bxdfId, flags);
    }

    __host__ void Host::Material::Bind(RenderObjectContainer& objectContainer)
    {
        // Push the binding to the device
        AssetHandle<Host::BxDF> bxdfAsset = GetAssetHandleForBinding<Host::Material, Host::BxDF>(objectContainer, m_bxdfId);
        if (bxdfAsset)
        {
            SynchroniseObjects(GetDeviceInstance(), bxdfAsset->GetDeviceInstance());
        }
    }
}
    