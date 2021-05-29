#pragma once

#include "generic/StdIncludes.h"

namespace Cuda
{

    class AssetBase
    {
    private:
        std::string     m_assetName;

    protected:
        template<typename T/*, typename = std::enable_if<std::is_base_of<AssetBase, T>::value>::type*/> friend class Asset;

        AssetBase() = default;
        AssetBase(const std::string& name) : m_assetName(name) {}

        virtual void OnDestroyAsset() = 0;
        void SetAssetName(const std::string& name) { m_assetName = name; }

    public:
        const std::string& GetAssetName() const { return m_assetName; }
    };

    class GlobalAssetRegistry
    {
    public:
        static GlobalAssetRegistry& Get();

        void Register(std::shared_ptr<AssetBase> object);
        void Deregister(std::shared_ptr<AssetBase> object);
        void VerifyEmpty();

    private:
        GlobalAssetRegistry() = default;

        std::unordered_map<AssetBase*, std::weak_ptr<AssetBase>>      m_assetMap;
        std::mutex                                                    m_mutex;
    };

    template<typename T/*, typename = std::enable_if<std::is_base_of<AssetBase, T>::value>::type*/>
    class Asset
    {
    private:
        std::shared_ptr<T>          m_ptr;

    public:
        Asset() = default;
        ~Asset() = default;

        template<typename... Pack>
        Asset(const std::string& assetName, Pack... args)
        {
            m_ptr.reset(new T(args...));
            m_ptr->SetAssetName(assetName);

            GlobalAssetRegistry::Get().Register(m_ptr);
        }

        Asset(T* ptr, const std::string& assetName)
        {
            m_ptr.reset(ptr);
            m_ptr->SetAssetName(assetName);

            GlobalAssetRegistry::Get().Register(m_ptr);
        }

        void DestroyAsset()
        {
            AssertMsgFmt(m_ptr.use_count() == 1, "Asset '%s' is still being referenced by %i other objects. Remove all other references before destroying this object.",
                m_ptr->GetAssetName().c_str(), m_ptr.use_count() - 1);

            std::printf("Destroyed '%s' with %i counts remaining.\n", m_ptr->GetAssetName().c_str(), m_ptr.use_count() - 1);

            m_ptr->OnDestroyAsset();
            GlobalAssetRegistry::Get().Deregister(m_ptr);
            m_ptr.reset();
        }

        inline operator bool() const { return m_ptr; }
        inline bool operator!() const { return !m_ptr; }

        inline T* operator->() { return &operator*(); }
        inline const T* operator->() const { return &operator*(); }

        inline T* get() { return m_ptr.get(); }
        inline const T* get() const { return m_ptr; }

        inline const T& operator*() const
        {
            return *m_ptr;
        }
        inline T& operator*()
        {
            return *m_ptr;
        }
    };

}