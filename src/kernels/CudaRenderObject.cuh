#pragma once

#include "math/CudaMath.cuh"
#include "CudaAsset.cuh"
#include <map>

namespace Cuda
{
    class RenderObjectContainer;

    enum RenderObjectFlags : uint 
    { 
        kRenderObjectDisabled = 1 << 0,
        kRenderObjectExcludeFromBake = 1 << 1,        
        kRenderObjectIsChild = 1 << 2,
        kRenderObjectUserFacingParameterMask = (kRenderObjectExcludeFromBake << 1) - 1
    };

    enum RenderObjectInstanceFlags : uint
    {
        kInstanceFlagsAllowMultipleInstances = 1 << 0,
        kInstanceSingleton = 1 << 2
    };

    struct RenderObjectParams
    {
        __host__ __device__ RenderObjectParams();
        __host__ RenderObjectParams(const ::Json::Node& node, const uint flags) : RenderObjectParams() { FromJson(node, flags);  }

        __host__ void FromJson(const ::Json::Node& node, const uint flags);
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
            __host__ virtual void           OnPreRenderPass(const float wallTime, const uint frameIdx) {}
            __host__ virtual void           OnPostRenderPass() {}
            __host__ virtual void           OnUpdateSceneGraph(RenderObjectContainer& sceneObjects) {}
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

        protected:
            __host__ RenderObject() : m_renderObjectFlags(0) {}
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
                    Log::Error("Unable to bind %s '%s' to %s '%s': asset is not the correct type.\n", BindType::GetAssetTypeString(), otherId, ThisType::GetAssetTypeString(), GetAssetID(), otherId);
                    return nullptr;
                }
              
                Log::Write("Bound %s '%s' to %s '%s'.\n", BindType::GetAssetTypeString(), otherId, ThisType::GetAssetTypeString(), GetAssetID());
                return downcastAsset;
            } 

        private:
            std::string         m_dagPath;
            uint                m_renderObjectFlags;
        };     
    }      
}