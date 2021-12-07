#pragma once

#include "generic/StdIncludes.h"
#include <memory>

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
        __host__ void FromJson(const Json::Node&, const uint) {}

        bool operator==(const NullParams&) const { return true; }
    };
    
    namespace Device
    { 
        class Asset { };
    }
    
    namespace Host
    {
        class Asset
        {
        private:
            std::string     m_assetId;

        protected:
            cudaStream_t    m_hostStream;
            template<typename T/*, typename = std::enable_if<std::is_base_of<AssetBase, T>::value>::type*/> friend class AssetHandle;

            __host__ Asset() : m_hostStream(0) { }
            __host__ Asset(const std::string& name) : m_assetId(name), m_hostStream(0) {  }

            __host__ virtual void OnDestroyAsset() = 0;
            __host__ void SetAssetID(const std::string& name) 
            { 
                m_assetId = name; 
            }

        public:
            virtual ~Asset() = default;

            __host__ virtual void FromJson(const ::Json::Node& jsonNode, const uint flags) {}
            __host__ virtual AssetType GetAssetType() const { return AssetType::kUnknown;  }
            __host__ const inline std::string& GetAssetID() const { return m_assetId; }
            __host__ void SetHostStream(cudaStream_t& hostStream) { m_hostStream = hostStream; }
            __host__ virtual void Synchronise() {}
        };
    }

    class GlobalAssetRegistry
    {
    public:
        static GlobalAssetRegistry& Get();

        void Register(std::shared_ptr<Host::Asset> object);
        void Deregister(std::shared_ptr<Host::Asset> object);
        void VerifyEmpty();
        void Report();
        bool Exists(const std::string& id) const;
        size_t Size() const { return m_assetMap.size(); }

    private:
        GlobalAssetRegistry() = default;

        std::unordered_map<std::string, std::weak_ptr<Host::Asset>>     m_assetMap;
        std::mutex                                                      m_mutex;
    };

    enum AssetHandleFlags : uint 
    { 
        kAssetForceDestroy = 1 << 0,
        kAssetExpectNoRefs = 1 << 1,
        kAssetCleanupPass = 1 << 2,
        kAssetAssertOnError = 1 << 3
    };

    // FIXME: Integrate weak asset handles into their own class
    template<typename T> using WeakAssetHandle = std::weak_ptr<T>;

    template<typename T/*, typename = std::enable_if<std::is_base_of<AssetBase, T>::value>::type*/>
    class AssetHandle
    {
        template<typename T> friend class AssetHandle;
    private:
        std::shared_ptr<T>          m_ptr;  

        explicit AssetHandle(std::shared_ptr<T>& ptr) : m_ptr(ptr) {}

    public:
        AssetHandle() = default;
        AssetHandle(const std::nullptr_t&) {}
        ~AssetHandle() = default;

        template<typename OtherType>
        explicit AssetHandle(AssetHandle<OtherType>& other)
        {
            m_ptr = other.m_ptr;
        }

        template<typename OtherType>
        explicit AssetHandle(const WeakAssetHandle<OtherType>& weakHandle)
        {
            // FIXME: Const casting here to get around the fact that AssetHandle sheilds us from const traits whereas std::weak_ptr does not
            AssertMsg(!weakHandle.expired(), "Trying to a convert an expired weak asset handle to a strong one.");
            m_ptr = weakHandle.lock();
        }

        template<typename... Pack>
        AssetHandle(const std::string& assetName, Pack... args)
        {
            m_ptr.reset(new T(args...));
            m_ptr->SetAssetID(assetName);

            GlobalAssetRegistry::Get().Register(m_ptr);
        }

        template<typename Type, typename = typename std::enable_if<std::is_base_of<Host::Asset, Type>::value>::type>
        AssetHandle(Type* ptr, const std::string& assetName)
        {
            m_ptr.reset(ptr);
            m_ptr->SetAssetID(assetName);

            GlobalAssetRegistry::Get().Register(m_ptr);
        }

        template<typename NewType>
        AssetHandle<NewType> DynamicCast() const
        {
            return AssetHandle<NewType>(std::dynamic_pointer_cast<NewType>(m_ptr));
        }

        template<typename NewType>
        AssetHandle<NewType> StaticCast() const
        {
            return AssetHandle<NewType>(std::static_pointer_cast<NewType>(m_ptr));
        }

        inline int GetReferenceCount() const
        {
            return m_ptr.use_count();
        }

        bool DestroyAsset(const uint flags = 0)
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

            m_ptr->OnDestroyAsset();
            GlobalAssetRegistry::Get().Deregister(m_ptr);
            m_ptr.reset();

            Log::Debug("Destroyed '%s'.\n", assetId.c_str());

            return true;
        }

        inline operator bool() const { return bool(m_ptr); }
        inline bool operator!() const { return !m_ptr; }
        inline bool operator==(const AssetHandle& rhs) const { return m_ptr == rhs.m_ptr; }
        inline bool operator!=(const AssetHandle& rhs) const { return m_ptr != rhs.m_ptr; }

        inline T* operator->() { return &operator*(); }
        inline const T* operator->() const { return &operator*(); }

        inline T* get() { return m_ptr.get(); }
        inline const T* get() const { return m_ptr; }

        WeakAssetHandle<T> GetWeakHandle() const { return WeakAssetHandle<T>(m_ptr); }

        inline const T& operator*() const
        {
            if (!m_ptr)
            {
                AssertMsg(m_ptr, "Invalid asset handle");
            }
            return *m_ptr;
        }
        inline T& operator*()
        {
            if (!m_ptr)
            {
                AssertMsg(m_ptr, "Invalid asset handle");
            }
            return *m_ptr;
        }
    };

    __host__ inline GlobalAssetRegistry& AR() { return Cuda::GlobalAssetRegistry::Get(); }
}