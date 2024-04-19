#include "SceneContainer.cuh"
#include "../bih/BIH2DAsset.cuh"
#include "core/GenericObjectContainer.cuh"
#include "../lights/Light.cuh"
#include "../tracables/Tracable.cuh"
#include "../integrators/Camera.cuh"

namespace Enso
{
    __host__ Host::SceneContainer::SceneContainer(const Asset::InitCtx& initCtx) :
        Host::GenericObject(initCtx),
        m_allocator(*this)
    {
        cu_deviceInstance = m_allocator.InstantiateOnDevice<Device::SceneContainer>();
        
        m_hostGenericObjects = Host::AssetAllocator::CreateAsset<Host::GenericObjectContainer>(":gi2d/sceneObjects");
        
        m_hostTracables = m_allocator.CreateChildAsset<Host::TracableContainer>("tracables", kVectorHostAlloc);
        m_hostLights = m_allocator.CreateChildAsset<Host::LightContainer>("lights", kVectorHostAlloc);
        m_hostCameras = m_allocator.CreateChildAsset<Host::CameraContainer>("cameras", kVectorHostAlloc);
        m_hostSceneObjects = m_allocator.CreateChildAsset<Host::SceneObjectContainer>("widgets", kVectorHostAlloc);

        m_hostTracableBIH = m_allocator.CreateChildAsset<Host::BIH2DAsset>("tracablebih", 1);
        m_hostSceneBIH = m_allocator.CreateChildAsset<Host::BIH2DAsset>("widgetbih", 1);

        m_deviceObjects.tracables = m_hostTracables->GetDeviceInstance();
        m_deviceObjects.lights = m_hostLights->GetDeviceInstance();
        m_deviceObjects.sceneObjects = m_hostSceneObjects->GetDeviceInstance();
        m_deviceObjects.tracableBIH = m_hostTracableBIH->GetDeviceInstance();
        m_deviceObjects.sceneBIH = m_hostSceneBIH->GetDeviceInstance();

        SynchroniseObjects<Device::SceneContainer>(cu_deviceInstance, m_deviceObjects);
    }

    __host__ Host::SceneContainer::~SceneContainer() noexcept
    {
        m_allocator.DestroyOnDevice(cu_deviceInstance);

        m_hostTracables.DestroyAsset();
        m_hostLights.DestroyAsset();
        m_hostCameras.DestroyAsset();
        m_hostTracableBIH.DestroyAsset();
        m_hostSceneBIH.DestroyAsset();

        m_hostGenericObjects.DestroyAsset();
    }   

    __host__ void Host::SceneContainer::Destroy()
    {
        if (m_hostGenericObjects)
        {
            for (auto& object : *m_hostGenericObjects)
            {
                object.DestroyAsset();
            }
        }
    }

    __host__ void Host::SceneContainer::Emplace(AssetHandle<Host::GenericObject>& newObject)
    {
        m_hostGenericObjects->Emplace(newObject);
    }

    __host__ void Host::SceneContainer::Clear()
    {
        m_hostTracables->Clear();
        m_hostLights->Clear();
        m_hostCameras->Clear();
        m_hostSceneObjects->Clear();
    }

    __host__ void Host::SceneContainer::Synchronise(const uint flags)
    {
        // Sync the scene objects with the device
        m_hostTracables->Synchronise(kVectorSyncUpload);
        m_hostLights->Synchronise(kVectorSyncUpload);
        m_hostCameras->Synchronise(kVectorSyncUpload);
        m_hostSceneObjects->Synchronise(kVectorSyncUpload);
    }

    __host__ void Host::SceneContainer::Summarise() const
    {
        Log::Write("%i scene objects", m_hostSceneObjects->Size());
        Log::Write("%i cameras", m_hostCameras->Size());
        Log::Write("Tracable BIH: %s", m_hostTracableBIH->GetBoundingBox().Format());
        Log::Write("Scene BIH: %s", m_hostSceneBIH->GetBoundingBox().Format());
    }

    __host__ bool Host::SceneContainer::Serialise(Json::Node& rootNode, const int flags) const
    {
        // TODO: Move serialisation code out of GI2DRenderer to here
        return true;
    }
}