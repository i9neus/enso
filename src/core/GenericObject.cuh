#pragma once

#include "math/Math.cuh"
#include "io/Serialisable.cuh"
#include "DirtinessGraph.cuh"

#include <map>
#include <unordered_set>
#include <functional>

namespace Enso
{
    template<typename FlagType>
    __host__ __inline__ bool SetGenericFlags(FlagType& data, const FlagType& newFlags, const bool isSet)
    {
        const FlagType prevData = data;
        if (isSet) { data |= newFlags; }
        else { data &= ~newFlags; }

        return prevData != data;
    }
    
    class GenericObjectContainer;

    enum AssetSyncType : int { kSyncObjects = 1, kSyncParams = 2 };

    enum GenericObjectFlags : uint
    {
        kGenericObjectDisabled = 1u << 0,
        kGenericObjectExcludeFromBake = 1u << 1,
        kGenericObjectIsChild = 1u << 2,
        kGenericObjectUserFacingParameterMask = (kGenericObjectExcludeFromBake << 1) - 1u
    };

    enum GenericObjectInstanceFlags : uint
    {
        kInstanceFlagsAllowMultipleInstances = 1u << 0,
        kInstanceSingleton = 1u << 2
    };

    enum GenericObjectDirtyFlags : uint
    {
        kGenericObjectClean = 0u,
        kGenericObjectDirtyRender = 1u << 1,
        kGenericObjectDirtyProbeGrids = 1u << 2,
        kGenericObjectDirtyAll = kGenericObjectDirtyRender | kGenericObjectDirtyProbeGrids
    };

    struct GenericObjectParams
    {
        __host__ __device__ GenericObjectParams();
        __host__ GenericObjectParams(const Json::Node& node, const uint flags) : GenericObjectParams() { FromJson(node, flags);  }

        __host__ uint FromJson(const Json::Node& node, const uint flags);
        __host__ void ToJson(Json::Node& node) const;
        __host__ void Randomise(const vec2& range);
    };
    
    namespace Device
    {
        class GenericObject : public Device::Asset
        {
        public:
            __device__ GenericObject() {}
            __device__ virtual ~GenericObject() {}

            __host__ __device__ virtual void    OnSynchronise(const int) {}
        };
    }

    namespace Host
    {        
        class SceneContainer;
        
        class GenericObject : public Host::Dirtyable,
                              public Serialisable
        {
        public:            
            __host__ virtual void Bind() {}
            
            __host__ virtual std::vector<AssetHandle<Host::GenericObject>>  GetChildObjectHandles() { return std::vector<AssetHandle<Host::GenericObject>>();  }
            __host__ virtual const GenericObjectParams*                     GetGenericObjectParams() const { return nullptr; }
            
            __host__ void                   UpdateDAGPath(const Json::Node& node);
            __host__ const std::string&     GetDAGPath() const { return m_dagPath; }
            __host__ bool                   HasDAGPath() const { return !m_dagPath.empty(); }

            __host__ bool                   IsChildObject() const { return m_renderObjectFlags & kGenericObjectIsChild; }
            __host__ static uint            GetInstanceFlags() { return 0; }

            __host__ virtual void           OnPreRender() {}
            __host__ virtual void           OnPostRender() {}
            __host__ virtual void           OnPreRenderPass(const float wallTime) {}
            __host__ virtual void           OnPostRenderPass() {}
            __host__ virtual void           OnUpdateSceneGraph(GenericObjectContainer& sceneObjects, const uint dirtyFlags) {}
            __host__ virtual bool           EmitStatistics(Json::Node& node) const { return false; }
            __host__ virtual void           Synchronise(const uint flags) {}

            __host__ void SetDAGPath(const std::string& dagPath) { m_dagPath = dagPath; }
            __host__ void SetGenericObjectFlags(const uint flags, const bool set = true)
            {
                if (set) { m_renderObjectFlags |= flags; }
                else { m_renderObjectFlags &= ~flags; }
            }

            __host__ void SetUserFacingGenericObjectFlags(const uint flags)
            {
                m_renderObjectFlags = (m_renderObjectFlags & ~kGenericObjectUserFacingParameterMask) | 
                                        (flags & kGenericObjectUserFacingParameterMask);
            }

            template<typename Subclass, typename DeligateLambda>
            __host__ void Listen(Subclass& owner, const std::string& eventID, DeligateLambda deligate)
            {
                for (auto it = m_actionDeligates.find(eventID); it != m_actionDeligates.end() && it->first == eventID; ++it)
                {
                    if (&it->second.m_owner == static_cast<GenericObject*>(&owner))
                    {
                        Log::Error("Internal error: deligate '%s' ['%s' -> '%s'] is already registered", eventID, GetAssetID(), owner.GetAssetID());
                        return;
                    }
                }

                m_actionDeligates.emplace(eventID, EventDeligate(owner,
                    std::function<void(const GenericObject&, const std::string&)>(std::bind(deligate, &owner, std::placeholders::_1, std::placeholders::_2))));
            }

            __host__ void  Unlisten(const GenericObject& owner, const std::string& eventID);

        protected:
            __host__ GenericObject(const Asset::InitCtx& initCtx);
            __host__ virtual ~GenericObject() noexcept {}
            __host__ void SetDeviceInstance(Device::GenericObject* deviceInstance);

            template<typename ThisType, typename BindType>
            __host__ AssetHandle<BindType> GetAssetHandleForBinding(GenericObjectContainer& objectContainer, const std::string& otherId)
            {
                // Try to find a handle to the asset
                AssetHandle<GenericObject> baseAsset = objectContainer.FindByID(otherId);
                if (!baseAsset)
                {
                    Log::Error("Unable to bind %s '%s' to %s '%s': %s does not exist.\n", BindType::GetAssetClassStatic(), otherId, ThisType::GetAssetClassStatic(), GetAssetID(), otherId);
                    return nullptr;
                }

                // Try to downcast it
                AssetHandle<BindType> downcastAsset = baseAsset.DynamicCast<BindType>();
                if (!downcastAsset)
                {
                    Log::Error("Unable to bind %s '%s' to %s '%s': asset is not  he correct type.\n", BindType::GetAssetClassStatic(), otherId, ThisType::GetAssetClassStatic(), GetAssetID(), otherId);
                    return nullptr;
                }
              
                Log::Write("Bound %s '%s' to %s '%s'.\n", BindType::GetAssetClassStatic(), otherId, ThisType::GetAssetClassStatic(), GetAssetID());
                return downcastAsset;
            } 

            __host__ void                   OnEvent(const std::string& eventID);
            __host__ void                   RegisterEvent(const std::string& eventID);

            __host__ uint                   SetDirtyFlags(const uint flags, const bool isSet = true) { SetGenericFlags(m_dirtyFlags, flags, isSet); return m_dirtyFlags; }
            __host__ void                   ClearDirtyFlags() { m_dirtyFlags = 0; }

        protected:    
            uint                m_dirtyFlags;
            bool                m_isFinalised;
            bool                m_isConstructed;

            const AssetAllocator m_allocator;

        private:
            std::string             m_dagPath;
            uint                    m_renderObjectFlags;
            Device::GenericObject*  cu_deviceInstance;

            struct EventDeligate
            {
                EventDeligate(GenericObject& owner, std::function<void(const GenericObject&, const std::string&)>& functor) :
                    m_owner(owner),
                    m_functor(functor) {}

                GenericObject&                                                   m_owner;
                std::function<void(const GenericObject&, const std::string&)>    m_functor;

                bool operator <(const EventDeligate& rhs)
                {
                    return std::hash<GenericObject*>{}(&m_owner) < std::hash<GenericObject*>{}(&rhs.m_owner);
                }
            };

            using EventDeligateMap = std::multimap <std::string, EventDeligate>;
            std::unordered_set<std::string>     m_eventRegistry;
            EventDeligateMap                    m_actionDeligates;

        };     
    }      
}