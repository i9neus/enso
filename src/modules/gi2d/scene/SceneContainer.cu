#include "SceneContainer.cuh"
#include "../bih/BIH2DAsset.cuh"
#include "core/GenericObjectContainer.cuh"
#include "../lights/Light.cuh"
#include "../tracables/Tracable.cuh"
#include "../integrators/Camera.cuh"

namespace Enso
{
    __host__ Host::SceneContainer::SceneContainer(const Asset::InitCtx& initCtx) :
        Host::GenericObject(initCtx)
    {
        cu_deviceInstance = AssetAllocator::InstantiateOnDevice<Device::SceneContainer>(*this);
        
        m_hostGenericObjects = AssetAllocator::CreateChildAsset<Host::GenericObjectContainer>(*this, ":gi2d/sceneObjects");
        
        m_hostTracables = AssetAllocator::CreateChildAsset<Host::TracableContainer>(*this, "tracables", kVectorHostAlloc);
        m_hostLights = AssetAllocator::CreateChildAsset<Host::LightContainer>(*this, "lights", kVectorHostAlloc);
        m_hostCameras = AssetAllocator::CreateChildAsset<Host::CameraContainer>(*this, "cameras", kVectorHostAlloc);
        m_hostSceneObjects = AssetAllocator::CreateChildAsset<Host::SceneObjectContainer>(*this, "widgets", kVectorHostAlloc);

        m_hostTracableBIH = AssetAllocator::CreateChildAsset<Host::BIH2DAsset>(*this, "tracablebih", 3);
        m_hostSceneBIH = AssetAllocator::CreateChildAsset<Host::BIH2DAsset>(*this, "widgetbih", 3);

        m_deviceObjects.tracables = m_hostTracables->GetDeviceInstance();
        m_deviceObjects.lights = m_hostLights->GetDeviceInstance();
        m_deviceObjects.sceneObjects = m_hostSceneObjects->GetDeviceInstance();
        m_deviceObjects.tracableBIH = m_hostTracableBIH->GetDeviceInstance();
        m_deviceObjects.sceneBIH = m_hostSceneBIH->GetDeviceInstance();

        SynchroniseObjects<Device::SceneContainer>(cu_deviceInstance, m_deviceObjects);
    }

    __host__ Host::SceneContainer::~SceneContainer() noexcept
    {
        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);

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

    __host__ void Host::SceneContainer::Prepare()
    {
        Assert(m_hostSceneObjects);        
        for (auto& object : *m_hostSceneObjects)
        {
            object->Prepare();
        }
    }

    __host__ void Host::SceneContainer::Clean()
    {
        Assert(m_hostGenericObjects);
        for (auto& object : *m_hostGenericObjects)
        {
            object->Clean();
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
        m_hostTracables->Synchronise(kVectorSyncUpload);
        m_hostLights->Synchronise(kVectorSyncUpload);
        m_hostCameras->Synchronise(kVectorSyncUpload);
        m_hostSceneObjects->Synchronise(kVectorSyncUpload);
    }

    __host__ void Host::SceneContainer::Summarise() const
    {
        Log::Indent("Rebuilt scene:");
        Log::Debug("%i scene objects", m_hostSceneObjects->Size());
        Log::Debug("%i cameras", m_hostCameras->Size());
        Log::Debug("Tracable BIH: %s", m_hostTracableBIH->GetBoundingBox().Format());
        Log::Debug("Scene BIH: %s", m_hostSceneBIH->GetBoundingBox().Format());
    }

    __host__ bool Host::SceneContainer::Serialise(Json::Node& rootNode, const int flags) const
    {
        // TODO: Move serialisation code out of GI2DRenderer to here
        return true;
    }
}