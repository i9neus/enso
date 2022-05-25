#pragma once

#include "math/CudaMath.cuh"
#include "CudaAsset.cuh"

#include <map>
#include <unordered_set>

namespace Cuda
{
    class RenderObjectContainer;

    enum RenderObjectFlags : uint 
    { 
        kRenderObjectDisabled                   = 1u << 0,
        kRenderObjectExcludeFromBake            = 1u << 1,        
        kRenderObjectIsChild                    = 1u << 2,
        kRenderObjectUserFacingParameterMask    = (kRenderObjectExcludeFromBake << 1) - 1u
    };

    enum RenderObjectInstanceFlags : uint
    {
        kInstanceFlagsAllowMultipleInstances    = 1u << 0,
        kInstanceSingleton                      = 1u << 2
    };

    enum RenderObjectDirtyFlags : uint
    {
        kRenderObjectClean                      = 0u,
        kRenderObjectDirtyRender                = 1u << 1,
        kRenderObjectDirtyProbeGrids            = 1u << 2,
        kRenderObjectDirtyAll                   = kRenderObjectDirtyRender | kRenderObjectDirtyProbeGrids
    };

    struct RenderObjectParams
    {
        __host__ __device__ RenderObjectParams();
        __host__ RenderObjectParams(const ::Json::Node& node, const uint flags) : RenderObjectParams() { FromJson(node, flags);  }

        __host__ uint FromJson(const ::Json::Node& node, const uint flags);
        __host__ void ToJson(::Json::Node& node) const;
        __host__ void Randomise(const vec2& range);

        JitterableFlags     flags;
    };
    
    namespace Device
    {
        class RenderObject : public Device::Asset
        {
        public:
            __device__ RenderObject() {}
            __device__ virtual ~RenderObject() {}
        };
    }

    namespace Host
    {        
        class RenderObject : public Host::Asset
        {
        public:            
            __host__ virtual void                                           Bind(RenderObjectContainer& objectContainer) {}
            __host__ virtual std::vector<AssetHandle<Host::RenderObject>>   GetChildObjectHandles() { return std::vector<AssetHandle<Host::RenderObject>>();  }
            __host__ virtual const RenderObjectParams*                      GetRenderObjectParams() const { return nullptr; }
            
            __host__ void                   UpdateDAGPath(const ::Json::Node& node);
            __host__ const std::string&     GetDAGPath() const { return m_dagPath; }
            __host__ const bool             HasDAGPath() const { return !m_dagPath.empty(); }

            __host__ bool                   IsChildObject() const { return m_renderObjectFlags & kRenderObjectIsChild; }
            __host__ static uint            GetInstanceFlags() { return 0; }

            __host__ virtual void           OnPreRender() {}
            __host__ virtual void           OnPostRender() {}
            __host__ virtual void           OnPreRenderPass(const float wallTime) {}
            __host__ virtual void           OnPostRenderPass() {}
            __host__ virtual void           OnUpdateSceneGraph(RenderObjectContainer& sceneObjects, const uint dirtyFlags) {}
            __host__ virtual bool           EmitStatistics(Json::Node& node) const { return false; }

            __host__ void SetDAGPath(const std::string& dagPath) { m_dagPath = dagPath; }
            __host__ void SetRenderObjectFlags(const uint flags, const bool set = true)
            {
                if (set) { m_renderObjectFlags |= flags; }
                else { m_renderObjectFlags &= ~flags; }
            }

            __host__ void SetUserFacingRenderObjectFlags(const uint flags)
            {
                m_renderObjectFlags = (m_renderObjectFlags & ~kRenderObjectUserFacingParameterMask) | 
                                        (flags & kRenderObjectUserFacingParameterMask);
            }

            template<typename Subclass, typename DeligateLambda>
            __host__ void Listen(Subclass& owner, const std::string& eventID, DeligateLambda deligate)
            {
                for (auto it = m_actionDeligates.find(eventID); it != m_actionDeligates.end() && it->first == eventID; ++it)
                {
                    if (&it->second.m_owner == static_cast<RenderObject*>(&owner))
                    {
                        Log::Error("Internal error: deligate '%s' ['%s' -> '%s'] is already registered", eventID, GetAssetID(), owner.GetAssetID());
                        return;
                    }
                }

                m_actionDeligates.emplace(eventID, EventDeligate(owner,
                    std::function<void(const RenderObject&, const std::string&)>(std::bind(deligate, &owner, std::placeholders::_1, std::placeholders::_2))));
            }

            __host__ void  Unlisten(const RenderObject& owner, const std::string& eventID);

        protected:
            __host__ RenderObject(const std::string& id) : Asset(id), m_renderObjectFlags(0) {}
            __host__ virtual ~RenderObject() = default; 

            template<typename ThisType, typename BindType>
            __host__ AssetHandle<BindType> GetAssetHandleForBinding(RenderObjectContainer& objectContainer, const std::string& otherId)
            {
                // Try to find a handle to the asset
                AssetHandle<RenderObject> baseAsset = objectContainer.FindByID(otherId);
                if (!baseAsset)
                {
                    Log::Error("Unable to bind %s '%s' to %s '%s': %s does not exist.\n", BindType::GetAssetTypeString(), otherId, ThisType::GetAssetTypeString(), GetAssetID(), otherId);
                    return nullptr;
                }

                // Try to downcast it
                AssetHandle<BindType> downcastAsset = baseAsset.DynamicCast<BindType>();
                if (!downcastAsset)
                {
                    Log::Error("Unable to bind %s '%s' to %s '%s': asset is not  he correct type.\n", BindType::GetAssetTypeString(), otherId, ThisType::GetAssetTypeString(), GetAssetID(), otherId);
                    return nullptr;
                }
              
                Log::Write("Bound %s '%s' to %s '%s'.\n", BindType::GetAssetTypeString(), otherId, ThisType::GetAssetTypeString(), GetAssetID());
                return downcastAsset;
            } 

            __host__ void                   OnEvent(const std::string& eventID);
            __host__ void                   RegisterEvent(const std::string& eventID);

        private:
            std::string         m_dagPath;
            uint                m_renderObjectFlags;

            struct EventDeligate
            {
                EventDeligate(RenderObject& owner, std::function<void(const RenderObject&, const std::string&)>& functor) :
                    m_owner(owner),
                    m_functor(functor) {}

                RenderObject&                                                   m_owner;
                std::function<void(const RenderObject&, const std::string&)>    m_functor;

                bool operator <(const EventDeligate& rhs)
                {
                    return std::hash<RenderObject*>{}(&m_owner) < std::hash<RenderObject*>{}(&rhs.m_owner);
                }
            };

            using EventDeligateMap = std::multimap <std::string, EventDeligate>;
            std::unordered_set<std::string>     m_eventRegistry;
            EventDeligateMap                    m_actionDeligates;

        };     
    }      
}