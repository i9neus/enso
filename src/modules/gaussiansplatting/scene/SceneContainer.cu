#include "SceneContainer.cuh"
#include "core/2d/SceneObject.cuh"
#include "core/2d/bih/BIH2DAsset.cuh"
#include "core/GenericObjectContainer.cuh"

namespace Enso
{
    __host__ Host::SceneContainer::SceneContainer(const Asset::InitCtx& initCtx) :
        Host::GenericObject(initCtx)
    {
        cu_deviceInstance = AssetAllocator::InstantiateOnDevice<Device::SceneContainer>(*this);        
        m_hostGenericObjects = AssetAllocator::CreateChildAsset<Host::GenericObjectContainer>(*this, ":gaussiansplatting/genericObjects"); 
        m_hostSceneObjects = AssetAllocator::CreateChildAsset<Host::SceneObjectContainer>(*this, "widgets", kVectorHostAlloc);
        m_hostSceneBIH = AssetAllocator::CreateChildAsset<Host::BIH2DAsset>(*this, "widgetbih", 3);

        m_deviceObjects.sceneObjects = m_hostSceneObjects->GetDeviceInstance();
        m_deviceObjects.sceneBIH = m_hostSceneBIH->GetDeviceInstance();

        SynchroniseObjects<Device::SceneContainer>(cu_deviceInstance, m_deviceObjects);
    }

    __host__ Host::SceneContainer::~SceneContainer() noexcept
    {
        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);

        m_hostSceneBIH.DestroyAsset();
        m_hostSceneObjects.DestroyAsset();
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
            //object->Prepare();
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
        m_hostSceneObjects->Clear();
    }

    __host__ void Host::SceneContainer::Synchronise(const uint flags)
    {       
        m_hostSceneObjects->Synchronise(kVectorSyncUpload);
    }

    __host__ void Host::SceneContainer::Summarise() const
    {
        Log::Indent("Rebuilt scene:");
        Log::Debug("%i scene objects", m_hostSceneObjects->Size());
        Log::Debug("Scene BIH: %s", m_hostSceneBIH->GetBoundingBox().Format());
    }

    __host__ bool Host::SceneContainer::Serialise(Json::Node& rootNode, const int flags) const
    {
        // TODO: Move serialisation code out of GI2DModule to here
        return true;
    }
}