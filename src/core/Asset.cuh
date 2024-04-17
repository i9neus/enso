#pragma once

#include "GlobalResourceRegistry.cuh"
#include <memory>
#include "io/Log.h"
#include "CudaHeaders.cuh"

namespace Enso
{
    namespace Json { class Node; }
    
    template<typename T/*, typename = std::enable_if<std::is_base_of<AssetBase, T>::value>::type*/>  class AssetHandle;

    template<typename HostType, typename DeviceType>
    class AssetTags
    {
    public:
        using DeviceVariant = DeviceType;
        using HostVariant = HostType;
    };

    // FIXME: Integrate weak asset handles into their own class
    template<typename AssetType> class AssetHandle;
    template<typename T> using WeakAssetHandle = std::weak_ptr<T>;
    
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

            // Weak references to the shared handles that own this instance
            WeakAssetHandle<Asset>  m_thisAssetHandle;
            WeakAssetHandle<Asset>  m_parentAssetHandle;

        public:
            __host__ virtual ~Asset() {}

            __host__ virtual uint               FromJson(const Json::Node& jsonNode, const uint flags) { return 0u; }
            __host__ virtual std::string        GetAssetClass() const { return ""; }
            __host__ const inline std::string&  GetAssetID() const { return m_assetId; }
            __host__ std::string                GetParentAssetID() const;
            __host__ void                       SetHostStream(cudaStream_t& hostStream) { m_hostStream = hostStream; }

            __host__ const WeakAssetHandle<Asset>& GetAssetHandle() const { return m_thisAssetHandle; }
            __host__ const WeakAssetHandle<Asset>& GetParentAssetHandle() const { return m_parentAssetHandle; }

            __host__ static std::string         MakeTemporaryID();

        protected:
            cudaStream_t                        m_hostStream;
            template<typename AssetType> friend class AssetHandle;
            
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

    template<typename AssetType/*, typename = std::enable_if<std::is_base_of<AssetBase, T>::value>::type*/>
    class AssetHandle
    {
        friend class Host::Asset;
        friend class Host::AssetAllocator;

        template<typename T> friend class AssetHandle;
        //template<typename AssetType, typename... Args> friend AssetHandle<AssetType> CreateAsset(const std::string&, Args...);

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

            // Delete then deregister the host object.
            m_ptr.reset();
            GlobalResourceRegistry::Get().DeregisterAsset(assetId);

            Log::Debug("Destroyed '%s'.\n", assetId);
            return true;
        }
    };
}