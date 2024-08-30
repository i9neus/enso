#include "SceneContainer.cuh"

#include "tracables/Tracable.cuh"
#include "lights/LightSampler.cuh"
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
        m_hostLightSamplers = AssetAllocator::CreateChildAsset<Host::LightSamplerContainer>(*this, "lightscontainer");
        m_hostMaterials = AssetAllocator::CreateChildAsset<Host::MaterialContainer>(*this, "materialscontainer");
        m_hostTextures = AssetAllocator::CreateChildAsset<Host::Texture2DContainer>(*this, "texturescontainer");
        m_hostCameras = AssetAllocator::CreateChildAsset<Host::CameraContainer>(*this, "camerascontainer");

        m_deviceObjects.tracables = m_hostTracables->GetDeviceInstance();
        m_deviceObjects.lightSamplers = m_hostLightSamplers->GetDeviceInstance();
        m_deviceObjects.materials = m_hostMaterials->GetDeviceInstance();
        m_deviceObjects.textures = m_hostTextures->GetDeviceInstance();
        m_deviceObjects.cameras = m_hostCameras->GetDeviceInstance();

        SynchroniseObjects<Device::SceneContainer>(cu_deviceInstance, m_deviceObjects);
    }

    __host__ Host::SceneContainer::~SceneContainer() noexcept
    {
        DestroyManagedObjects();
        
        m_hostTracables.DestroyAsset();
        m_hostLightSamplers.DestroyAsset();
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
        m_sceneObjects.clear();
        
        DestroyObjects(*m_hostTracables);
        DestroyObjects(*m_hostLightSamplers);
        DestroyObjects(*m_hostMaterials);
        DestroyObjects(*m_hostTextures);
        DestroyObjects(*m_hostCameras);
    }

    template<typename ObjectType, typename ContainerType>
    __host__ void EmplaceSceneObject(AssetHandle<ObjectType> newObject, ContainerType& container, 
                                     std::unordered_map<std::string, AssetHandle<Host::SceneObject>>& sceneObjects,
                                     std::unordered_map<std::string, int>& containerIndices)
    {
        Assert(newObject);

        const std::string assetStem = newObject->GetAssetStem();
        AssertMsgFmt(sceneObjects.find(assetStem) == sceneObjects.end(), "Asset '%s' already in scene object list", assetStem);
        AssertMsgFmt(containerIndices.find(assetStem) == containerIndices.end(), "Asset '%s' already in scene object list", assetStem);

        sceneObjects.emplace(assetStem, AssetHandle<Host::SceneObject>(newObject));
        containerIndices.emplace(assetStem, container.size());
        container.push_back(newObject);

        Log::Debug("Added new object '%s'", assetStem);
    }

    __host__ void Host::SceneContainer::Emplace(AssetHandle<Host::Tracable> newTracable) { EmplaceSceneObject(newTracable, *m_hostTracables, m_sceneObjects, m_containerIndices); }
    __host__ void Host::SceneContainer::Emplace(AssetHandle<Host::LightSampler> newLight) { EmplaceSceneObject(newLight, *m_hostLightSamplers, m_sceneObjects, m_containerIndices); }
    __host__ void Host::SceneContainer::Emplace(AssetHandle<Host::Material> newMaterial) { EmplaceSceneObject(newMaterial, *m_hostMaterials, m_sceneObjects, m_containerIndices); }
    __host__ void Host::SceneContainer::Emplace(AssetHandle<Host::Texture2D> newTexture) { EmplaceSceneObject(newTexture, *m_hostTextures, m_sceneObjects, m_containerIndices); }
    __host__ void Host::SceneContainer::Emplace(AssetHandle<Host::Camera> newCamera) { EmplaceSceneObject(newCamera, *m_hostCameras, m_sceneObjects, m_containerIndices); }

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
        CleanContainerItems(m_hostLightSamplers);
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
            m_hostLightSamplers->Upload();
            m_hostCameras->Upload();
            m_hostMaterials->Upload();
            m_hostTextures->Upload();
        }
    }

    __host__ void Host::SceneContainer::Summarise() const
    {
        Log::Indent("Rebuilt scene:");
        Log::Debug("  - %i tracables", m_hostTracables->size());
        Log::Debug("  - %i light samplers", m_hostLightSamplers->size());
        Log::Debug("  - %i cameras", m_hostCameras->size());
        Log::Debug("  - %i materials", m_hostMaterials->size());
        Log::Debug("  - %i textures", m_hostTextures->size());
    }

    __host__ void Host::SceneContainer::SetEnvironmentTexture(const std::string& id)
    {
        auto obj = Find<Host::Texture2D>(id);
        if (!obj)
        {
            Log::Error("Error: couldn't set environment map to '%s'; texture not found.", id);
        }
        else
        {
            m_deviceObjects.envTexture = obj->GetDeviceInstance();
        }
    }
}