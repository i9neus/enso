#include "CudaEmitterMaterial.cuh"

namespace Cuda
{
    __host__ Host::EmitterMaterial::EmitterMaterial() :
        cu_deviceData(nullptr)
    {
        RenderObject::SetRenderObjectFlags(kIsChildObject);
        cu_deviceData = InstantiateOnDevice<Device::EmitterMaterial>();
    }

    __host__ void Host::EmitterMaterial::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceData);
    }

    __host__ void Host::EmitterMaterial::UpdateParams(const vec3& radiance)
    {
        SynchroniseObjects(cu_deviceData, radiance);
    }
}