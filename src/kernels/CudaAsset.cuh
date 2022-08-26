#pragma once

#include <generic/StdIncludes.h>
#include <memory>
#include "GlobalResourceRegistry.cuh"

namespace Json { class Node; }

namespace Cuda
{
    template<typename T/*, typename = std::enable_if<std::is_base_of<AssetBase, T>::value>::type*/>  class AssetHandle;

    template<typename HostType, typename DeviceType>
    class AssetTags
    {
    public:
        using DeviceVariant = DeviceType;
        using HostVariant = HostType;
    };

    enum class AssetType : int 
    { 
        kUnknown = -1, 
        kTracable, 
        kBxDF, 
        kMaterial, 
        kLight, 
        kCamera, 
        kIntegrator, 
        kLightProbeFilter
    };

    struct NullParams
    {
        __host__ __device__ NullParams() {}

        __host__ void ToJson(Json::Node&) const {}
        __host__ uint FromJson(const Json::Node&, const uint) { return 0u; }

        bool operator==(const NullParams&) const { return true; }
    };
    
    namespace Device
    { 
        class Asset {};
    }
    
    namespace Host
    {        
        class AssetAllocator;

        class Asset
        {
            friend class AssetAllocator;

        private:
            const std::string       m_assetId;
            std::string             m_parentAssetId;

        public:
            __host__ virtual ~Asset() {}

            __host__ virtual uint               FromJson(const ::Json::Node& jsonNode, const uint flags) { return 0u; }
            __host__ virtual AssetType          GetAssetType() const { return AssetType::kUnknown; }
            __host__ const inline std::string&  GetAssetID() const { return m_assetId; }
            __host__ const inline std::string&  GetParentAssetID() const { return m_parentAssetId; }
            __host__ void                       SetHostStream(cudaStream_t& hostStream) { m_hostStream = hostStream; }

            __host__ static std::string                  MakeTemporaryID();

            //template<typename AssetType, typename... Args>
            //__host__ static AssetHandle<AssetType> CreateAsset(const std::string& newId, Args... args);

        protected:
            cudaStream_t            m_hostStream;
            template<typename AssetType> friend class AssetHandle;
            
            __host__ virtual void OnDestroyAsset() = 0;

            template<typename AssetType, typename... Args>
            __host__ inline AssetHandle<AssetType> CreateChildAsset(const std::string& newId, Args... args);

        private:
            __host__ Asset(const std::string& id) : m_assetId(id), m_hostStream(0) {  }
        };
    }

    enum AssetHandleFlags : uint
    {
        kAssetForceDestroy = 1 << 0,
        kAssetExpectNoRefs = 1 << 1,
        kAssetCleanupPass = 1 << 2,
        kAssetAssertOnError = 1 << 3
    };

    // FIXME: Integrate weak asset handles into their own class
    template<typename T> using WeakAssetHandle = std::weak_ptr<T>;

    template<typename AssetType/*, typename = std::enable_if<std::is_base_of<AssetBase, T>::value>::type*/>
    class AssetHandle
    {
        friend class Host::Asset;
        template<typename T> friend class AssetHandle;
        template<typename AssetType, typename... Args> friend AssetHandle<AssetType> CreateAsset(const std::string&, Args...);

    private:
        std::shared_ptr<AssetType>          m_ptr;

        explicit AssetHandle(std::shared_ptr<AssetType>& ptr) : m_ptr(ptr) {}

    public:
        __host__ AssetHandle() {}
        __host__ AssetHandle(const std::nullptr_t&) {}
        __host__ ~AssetHandle() {}

        template<typename OtherType>
        __host__ AssetHandle(AssetHandle<OtherType>& other)
        {
            m_ptr = other.m_ptr;
        }

        template<typename OtherType>
        __host__ explicit AssetHandle(const WeakAssetHandle<OtherType>& weakHandle)
        {
            // FIXME: Const casting here to get around the fact that AssetHandle sheilds us from const traits whereas std::weak_ptr does not
            AssertMsg(!weakHandle.expired(), "Trying to a convert an expired weak asset handle to a strong one.");
            m_ptr = weakHandle.lock();
        }

        template<typename CastType>
        __host__ AssetHandle<CastType> DynamicCast() const
        {
            return AssetHandle<CastType>(std::dynamic_pointer_cast<CastType>(m_ptr));
        }

        template<typename CastType>
        __host__ AssetHandle<CastType> StaticCast() const
        {
            return AssetHandle<CastType>(std::static_pointer_cast<CastType>(m_ptr));
        }

        __host__ inline int GetReferenceCount() const
        {
            return m_ptr.use_count();
        }

        __host__ inline operator bool() const { return bool(m_ptr); }
        __host__ inline bool operator!() const { return !m_ptr; }
        __host__ inline bool operator==(const AssetHandle& rhs) const { return m_ptr == rhs.m_ptr; }
        __host__ inline bool operator!=(const AssetHandle& rhs) const { return m_ptr != rhs.m_ptr; }

        __host__ inline AssetType* operator->() { return &operator*(); }
        __host__ inline const AssetType* operator->() const { return &operator*(); }

        __host__ inline AssetType* get() { return m_ptr.get(); }
        __host__ inline const AssetType* get() const { return m_ptr; }

        __host__ WeakAssetHandle<AssetType> GetWeakHandle() const { return WeakAssetHandle<AssetType>(m_ptr); }

        __host__ inline const AssetType& operator*() const
        {
            if (!m_ptr)
            {
                AssertMsg(m_ptr, "Invalid asset handle");
            }
            return *m_ptr;
        }
        __host__ inline AssetType& operator*()
        {
            if (!m_ptr)
            {
                AssertMsg(m_ptr, "Invalid asset handle");
            }
            return *m_ptr;
        }

        __host__ bool DestroyAsset(const uint flags = 0)
        {
            if (!m_ptr) { return true; }

            // If the refcount is greater than 1, the object is still in use elsewhere
            if (m_ptr.use_count() > 1 && !(flags & kAssetForceDestroy))
            {
                if (flags & kAssetExpectNoRefs)
                {
                    Log::Error("Asset '%s' is still being referenced by %i other objects. Remove all other references before destroying this object.",
                        m_ptr->GetAssetID().c_str(), m_ptr.use_count() - 1);
                    Assert(!(flags & kAssetAssertOnError));
                }

                return false;
            }

            const std::string assetId = m_ptr->GetAssetID();

            // Notify the asset that it's about to be destroyed, then deregister and delete the host object.
            m_ptr->OnDestroyAsset();
            GlobalResourceRegistry::Get().DeregisterAsset(m_ptr, assetId);
            m_ptr.reset();

            Log::Debug("Destroyed '%s'.\n", assetId);
            return true;
        }
    };

    template<typename AssetType, typename... Args>
    __host__ inline AssetHandle<AssetType> Host::Asset::CreateChildAsset(const std::string& newId, Args... args)
    {
        static_assert(std::is_base_of<Host::Asset, AssetType>::value, "Asset type must be derived from Host::Asset");

        // Concatenate new asset ID with its parent ID 
        const std::string& concatId = GetAssetID() + "/" + newId;
        
        auto& registry = GlobalResourceRegistry::Get();
        AssertMsgFmt(!registry.Exists(concatId), "Object '%s' is already in asset registry!", newId.c_str());
        
        AssetHandle<AssetType> newAsset;
        newAsset.m_ptr = std::make_shared<AssetType>(concatId, args...);
        newAsset->m_parentAssetId = GetAssetID();

        registry.RegisterAsset(newAsset.m_ptr, concatId);
        return newAsset;
    }

    template<typename AssetType, typename... Args>
    __host__ inline AssetHandle<AssetType> CreateAsset(const std::string& newId, Args... args)
    {
        static_assert(std::is_base_of<Host::Asset, AssetType>::value, "Asset type must be derived from Host::Asset");

        auto& registry = GlobalResourceRegistry::Get();
        AssertMsgFmt(!registry.Exists(newId), "Object '%s' is already in asset registry!", newId.c_str());
        
        AssetHandle<AssetType> newAsset;
        newAsset.m_ptr = std::make_shared<AssetType>(newId, args...);

        registry.RegisterAsset(newAsset.m_ptr, newId);
        return newAsset;
    }
}