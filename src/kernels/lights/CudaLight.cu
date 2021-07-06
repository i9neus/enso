#include "CudaLight.cuh"

namespace Cuda
{
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