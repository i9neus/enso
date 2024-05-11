#pragma once

#include "math/Math.cuh"
#include "io/Serialisable.cuh"
#include "DirtinessGraph.cuh"
#include "AssetAllocator.cuh"

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
        };
    }

    namespace Host
    {        
        class SceneContainer;
        
        class GenericObject : public Host::Dirtyable,
                              public Serialisable
        {
        public:            
            __host__ void                   Bind() {}
            __host__ virtual void           Synchronise(const uint syncFlags) { }
            
            __host__ virtual std::vector<AssetHandle<Host::GenericObject>>  GetChildObjectHandles() { return std::vector<AssetHandle<Host::GenericObject>>();  }
            __host__ virtual const GenericObjectParams*                     GetGenericObjectParams() const { return nullptr; }
            
            __host__ bool                   IsChildObject() const { return m_genericObjectFlags & kGenericObjectIsChild; }
            __host__ static uint            GetInstanceFlags() { return 0; }

            __host__ virtual bool           EmitStatistics(Json::Node& node) const { return false; }
            //__host__ virtual void           Synchronise(const uint flags) {}

            __host__ void SetGenericObjectFlags(const uint flags, const bool set = true)
            {
                if (set) { m_genericObjectFlags |= flags; }
                else { m_genericObjectFlags &= ~flags; }
            }

            __host__ void SetUserFacingGenericObjectFlags(const uint flags)
            {
                m_genericObjectFlags = (m_genericObjectFlags & ~kGenericObjectUserFacingParameterMask) | 
                                        (flags & kGenericObjectUserFacingParameterMask);
            }          

        protected:
            __host__ GenericObject(const Asset::InitCtx& initCtx);
            __host__ virtual ~GenericObject() noexcept {}

            __host__ void                   SetDeviceInstance(Device::GenericObject* deviceInstance);

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

            //__host__ uint                   SetDirtyFlags(const uint flags, const bool isSet = true) { SetGenericFlags(m_dirtyFlags, flags, isSet); return m_dirtyFlags; }
            //__host__ void                   ClearDirtyFlags() { m_dirtyFlags = 0; }

        protected:    
            //uint                m_dirtyFlags;
            bool                m_isFinalised;
            bool                m_isConstructed;

        private:
            std::string             m_dagPath;
            uint                    m_genericObjectFlags;
        };     
    }      
}