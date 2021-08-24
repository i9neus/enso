#include "CudaEmitterMaterial.cuh"

namespace Cuda
{
    __host__ Host::EmitterMaterial::EmitterMaterial(const uint flags) :
        cu_deviceData(nullptr)
    {
        RenderObject::SetRenderObjectFlags(flags);
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