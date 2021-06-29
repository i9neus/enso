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

    enum class AssetType : int { kUnknown = -1, kTracable, kBxDF, kMaterial, kLight, kCamera, kIntegrator };
    
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
        size_t Size() const { return m_assetMap.size(); }

    private:
        GlobalAssetRegistry() = default;

        std::unordered_map<std::string, std::weak_ptr<Host::Asset>>     m_assetMap;
        std::mutex                                                      m_mutex;
    };

    enum AssetHandleFlags : uint { kAssetForceDestroy = 1, kAssetAssertOnError = 2 };

    template<typename T/*, typename = std::enable_if<std::is_base_of<AssetBase, T>::value>::type*/>
    class AssetHandle
    {
    private:
        std::shared_ptr<T>          m_ptr;     

    public:
        AssetHandle() = default;
        AssetHandle(const std::nullptr_t&) {}
        ~AssetHandle() = default;


        explicit AssetHandle(std::shared_ptr<T>& ptr) : m_ptr(ptr) {}

        template<typename OtherType>
        explicit AssetHandle(AssetHandle<OtherType>& other)
        {
            m_ptr = other.m_ptr;
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
        AssetHandle<NewType> DynamicCast()
        {
            AssertMsg(m_ptr, "Invalid asset handle");

            return AssetHandle<NewType>(std::dynamic_pointer_cast<NewType>(m_ptr));
        }

        bool DestroyAsset(const uint flags = 0)
        {
            if (!m_ptr) { return true; }
            
            // If the refcount is greater than 1, the object is still in use elsewhere
            if (m_ptr.use_count() > 1 && !(flags & kAssetForceDestroy))
            {
                AssertMsgFmt(!(flags & kAssetAssertOnError), "Asset '%s' is still being referenced by %i other objects. Remove all other references before destroying this object.",
                                 m_ptr->GetAssetID().c_str(), m_ptr.use_count() - 1);
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

        inline T* operator->() { return &operator*(); }
        inline const T* operator->() const { return &operator*(); }

        inline T* get() { return m_ptr.get(); }
        inline const T* get() const { return m_ptr; }

        inline const T& operator*() const
        {
            AssertMsg(m_ptr, "Invalid asset handle");
            return *m_ptr;
        }
        inline T& operator*()
        {
            AssertMsg(m_ptr, "Invalid asset handle");
            return *m_ptr;
        }
    };

    __host__ inline GlobalAssetRegistry& AR() { return Cuda::GlobalAssetRegistry::Get(); }
}