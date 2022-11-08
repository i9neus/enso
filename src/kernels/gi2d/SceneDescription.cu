#include "SceneDescription.cuh"
#include "BIH2DAsset.cuh"
#include "integrators/VoxelProxyGrid.cuh"

namespace GI2D
{
    __host__ Host::SceneDescription::SceneDescription(const std::string& id) :
        Cuda::Host::AssetAllocator(id)
    {
        cu_deviceInstance = InstantiateOnDevice<Device::SceneDescription>();
    }

    __host__ Host::SceneDescription::~SceneDescription()
    {
        DestroyOnDevice(cu_deviceInstance);
    }

    __host__ void Host::SceneDescription::Prepare()
    {
        if (sceneBIH) { m_deviceObjects.bih = sceneBIH->GetDeviceInstance(); }
        if (hostTracables) { m_deviceObjects.tracables = hostTracables->GetDeviceInstance(); }
        //if (hostInspectors) { m_deviceObjects. = hostInspectors->GetDeviceInstance(); }

        if (voxelProxy) { m_deviceObjects.voxelProxy = voxelProxy->GetDeviceInstance(); }

        SynchroniseInheritedClass<Device::SceneDescription>(cu_deviceInstance, m_deviceObjects, 0);
    }
}