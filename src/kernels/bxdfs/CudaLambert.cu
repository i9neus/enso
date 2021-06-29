#include "CudaLambert.cuh"

#include "generic/JsonUtils.h"

namespace Cuda
{
    __host__ AssetHandle<Host::RenderObject> Host::LambertBRDF::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kBxDF) { return AssetHandle<Host::RenderObject>(); }

        return AssetHandle<Host::RenderObject>(new Host::LambertBRDF(), id);
    }
    
    __host__ Host::LambertBRDF::LambertBRDF() :
        cu_deviceData(nullptr)
    {
        cu_deviceData = InstantiateOnDevice<Device::LambertBRDF>();
    }
    
    __host__ void Host::LambertBRDF::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceData);
    }
}