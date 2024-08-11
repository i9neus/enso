#include "SceneContainer.cuh"

#include "tracables/Tracable.cuh"
#include "lights/Light.cuh"
#include "materials/Material.cuh"
#include "textures/Texture2D.cuh"
#include "cameras/Camera.cuh"
#include "core/Vector.cuh"

namespace Enso
{
    __host__ Host::SceneContainer::SceneContainer(const Asset::InitCtx& initCtx) :
        Host::GenericObject(initCtx),
        cu_deviceInstance(AssetAllocator::InstantiateOnDevice<Device::SceneContainer>(*this))
    {
        m_hostTracables = AssetAllocator::CreateChildAsset<Host::TracableContainer>(*this, "tracablescontainer");
        m_hostLights = AssetAllocator::CreateChildAsset<Host::LightContainer>(*this, "lightscontainer");
        m_hostMaterials = AssetAllocator::CreateChildAsset<Host::MaterialContainer>(*this, "materialscontainer");
        m_hostTextures = AssetAllocator::CreateChildAsset<Host::Texture2DContainer>(*this, "texturescontainer");
        m_hostCameras = AssetAllocator::CreateChildAsset<Host::CameraContainer>(*this, "camerascontainer");

        m_deviceObjects.tracables = m_hostTracables->GetDeviceInstance();
        m_deviceObjects.lights = m_hostLights->GetDeviceInstance();
        m_deviceObjects.materials = m_hostMaterials->GetDeviceInstance();
        m_deviceObjects.textures = m_hostTextures->GetDeviceInstance();
        m_deviceObjects.cameras = m_hostCameras->GetDeviceInstance();

        SynchroniseObjects<Device::SceneContainer>(cu_deviceInstance, m_deviceObjects);
    }

    __host__ Host::SceneContainer::~SceneContainer() noexcept
    {
        DestroyManagedObjects();
        
        m_hostTracables.DestroyAsset();
        m_hostLights.DestroyAsset();
        m_hostMaterials.DestroyAsset();
        m_hostTextures.DestroyAsset();
        m_hostCameras.DestroyAsset();
        
        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);
    }   

    template<typename ContainerType> 
    __host__ void DestroyObjects(ContainerType& container)
    {
        for (auto& object : container)
        {
            object.DestroyAsset();
        }
    }

    __host__ void Host::SceneContainer::DestroyManagedObjects()
    {
        DestroyObjects(*m_hostTracables);
        DestroyObjects(*m_hostLights);
        DestroyObjects(*m_hostMaterials);
        DestroyObjects(*m_hostTextures);
        DestroyObjects(*m_hostCameras);
    }

    __host__ void Host::SceneContainer::Prepare()
    {
       
    }    

    __host__ void Host::SceneContainer::Synchronise(const uint flags)
    {
        m_hostTracables->Synchronise(kVectorSyncUpload);
        m_hostLights->Synchronise(kVectorSyncUpload);
        m_hostCameras->Synchronise(kVectorSyncUpload);
        m_hostMaterials->Synchronise(kVectorSyncUpload);
        m_hostTextures->Synchronise(kVectorSyncUpload);
    }

    __host__ void Host::SceneContainer::Summarise() const
    {
        Log::Indent("Rebuilt scene:");
        Log::Debug("  - %i tracables", m_hostTracables->size());
        Log::Debug("  - %i lights", m_hostLights->size());
        Log::Debug("  - %i cameras", m_hostCameras->size());
        Log::Debug("  - %i materials", m_hostMaterials->size());
        Log::Debug("  - %i textures", m_hostTextures->size());
    }

}