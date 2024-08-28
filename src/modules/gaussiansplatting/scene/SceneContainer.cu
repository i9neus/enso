#include "SceneContainer.cuh"

#include "tracables/Tracable.cuh"
#include "lights/Light.cuh"
#include "materials/Material.cuh"
#include "textures/Texture2D.cuh"
#include "cameras/Camera.cuh"
#include "core/containers/Vector.cuh"

namespace Enso
{
    __host__ Host::SceneContainer::SceneContainer(const Asset::InitCtx& initCtx) :
        Host::GenericObject(initCtx),
        cu_deviceInstance(AssetAllocator::InstantiateOnDevice<Device::SceneContainer>(*this))
    {
        Listen({ kDirtySceneObjectChanged });
        
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

    __host__ void Host::SceneContainer::OnDirty(const DirtinessEvent& flag, AssetHandle<Host::Asset>& caller)
    {
        if (flag == kDirtySceneObjectChanged)
        {
            auto sceneObj = caller.DynamicCast<Host::SceneObject>();
            if (sceneObj)
            {
                m_syncObjectSet[caller->GetAssetID()] = sceneObj.GetWeakHandle();
                SignalDirty(kDirtyParams);
            }
        }
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

    template<typename ContainerType>
    __host__ void CleanContainerItems(AssetHandle<ContainerType>& container)
    {
        Assert(container);
        for (auto& item : *container)
        {
            item->Clean();
        }
    }

    __host__ void Host::SceneContainer::OnClean()
    {
        CleanContainerItems(m_hostTracables);
        CleanContainerItems(m_hostLights);
        CleanContainerItems(m_hostMaterials);
        CleanContainerItems(m_hostTextures);
        CleanContainerItems(m_hostCameras);
    }

    __host__ void Host::SceneContainer::Synchronise(const uint flags)
    {
        // Synchronise the parameters of any objects that signalled they were dirty
        if (flags & kSyncParams)
        {
            for (auto& it : m_syncObjectSet)
            {
                AssetHandle<Host::SceneObject> object(it.second);
                if (object)
                {
                    object->Synchronise(kSyncParams);
                }
            }
            m_syncObjectSet.clear();
        }

        // Synchronise the scene objects arrays themselves
        if (flags & kSyncObjects)
        {
            m_hostTracables->Upload();
            m_hostLights->Upload();
            m_hostCameras->Upload();
            m_hostMaterials->Upload();
            m_hostTextures->Upload();
        }
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