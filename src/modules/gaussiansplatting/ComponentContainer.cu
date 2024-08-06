#include "ComponentContainer.cuh"
#include "core/2d/DrawableObject.cuh"
#include "core/2d/bih/BIH2DAsset.cuh"
#include "core/GenericObjectContainer.cuh"

namespace Enso
{
    __host__ Host::ComponentContainer::ComponentContainer(const Asset::InitCtx& initCtx) :
        Host::GenericObject(initCtx)
    {
        cu_deviceInstance = AssetAllocator::InstantiateOnDevice<Device::ComponentContainer>(*this);        
        m_hostGenericObjects = AssetAllocator::CreateChildAsset<Host::GenericObjectContainer>(*this, ":gaussiansplatting/genericObjects"); 
        m_hostDrawableObjects = AssetAllocator::CreateChildAsset<Host::DrawableObjectContainer>(*this, "drawables", kVectorHostAlloc);
        m_hostDrawableBIH = AssetAllocator::CreateChildAsset<Host::BIH2DAsset>(*this, "drawablebih", 3);

        m_deviceObjects.sceneObjects = m_hostDrawableObjects->GetDeviceInstance();
        m_deviceObjects.sceneBIH = m_hostDrawableBIH->GetDeviceInstance();

        SynchroniseObjects<Device::ComponentContainer>(cu_deviceInstance, m_deviceObjects);
    }

    __host__ Host::ComponentContainer::~ComponentContainer() noexcept
    {
        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);

        m_hostDrawableBIH.DestroyAsset();
        m_hostDrawableObjects.DestroyAsset();
        m_hostGenericObjects.DestroyAsset();        
    }   

    __host__ void Host::ComponentContainer::Destroy()
    {
        if (m_hostGenericObjects)
        {
            for (auto& object : *m_hostGenericObjects)
            {
                object.DestroyAsset();
            }
        }
    }

    __host__ void Host::ComponentContainer::Prepare()
    {
        Assert(m_hostDrawableObjects);        
        for (auto& object : *m_hostDrawableObjects)
        {
            //object->Prepare();
        }
    }

    __host__ void Host::ComponentContainer::Clean()
    {
        Assert(m_hostGenericObjects);
        for (auto& object : *m_hostGenericObjects)
        {
            object->Clean();
        }
    }

    __host__ void Host::ComponentContainer::Emplace(AssetHandle<Host::GenericObject>& newObject)
    {
        m_hostGenericObjects->Emplace(newObject);
    }

    __host__ void Host::ComponentContainer::Clear()
    {
        m_hostDrawableObjects->Clear();
    }

    __host__ void Host::ComponentContainer::Synchronise(const uint flags)
    {       
        m_hostDrawableObjects->Synchronise(kVectorSyncUpload);
    }

    __host__ void Host::ComponentContainer::Summarise() const
    {
        Log::Indent("Rebuilt scene:");
        Log::Debug("%i drawable objects", m_hostDrawableObjects->Size());
        Log::Debug("Drawable BIH: %s", m_hostDrawableBIH->GetBoundingBox().Format());
    }

    __host__ bool Host::ComponentContainer::Serialise(Json::Node& rootNode, const int flags) const
    {
        // TODO: Move serialisation code out of GI2DModule to here
        return true;
    }
}