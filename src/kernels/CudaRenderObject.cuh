#pragma once

#include "math/CudaMath.cuh"
#include "CudaAsset.cuh"
#include <map>

namespace Cuda
{
    class RenderObjectContainer;

    enum class RenderObjectContainerResult : uint { kSuccess = 0, kNotFound, kInvalidType };
    
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
            __host__ virtual void Bind(RenderObjectContainer& objectContainer) {}
            __host__ virtual void FromJson(const ::Json::Node& node, const uint flags) override;

            __host__ const std::string& GetDAGPath() const { return m_dagPath; }
            __host__ const bool HasDAGPath() const { return !m_dagPath.empty(); }

        protected:
            __host__ RenderObject() = default;
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

            __host__ void SetDAGPath(const std::string& dagPath) { m_dagPath = dagPath; }

        private:
            std::string m_dagPath;
        };     
    }  

    class RenderObjectContainer : public Host::Asset
    {
        using RenderObjectMap = std::map<std::string, AssetHandle<Host::RenderObject>>;

    private:
        RenderObjectMap       m_objectMap;
        RenderObjectMap       m_dagMap;

    public:
        class Iterator
        {
        private:
            RenderObjectMap::iterator m_it;
        public:
            Iterator(RenderObjectMap::iterator it) noexcept : m_it(it) {}

            Iterator& operator++() { ++m_it; return *this; }
            bool operator!=(const Iterator& rhs) const { return m_it != rhs.m_it; }
            AssetHandle<Host::RenderObject>& operator*() { return m_it->second; }
        };

        __host__ RenderObjectContainer() = default;
        __host__ virtual void OnDestroyAsset() override final;

        __host__ Iterator begin() noexcept { return Iterator(m_objectMap.begin()); }
        __host__ Iterator end() noexcept { return Iterator(m_objectMap.end()); }

        __host__ AssetHandle<Host::RenderObject> FindByID(const std::string& id)
        {
            auto it = m_objectMap.find(id);
            return (it == m_objectMap.end()) ? AssetHandle<Host::RenderObject>(nullptr) : it->second;
        }

        __host__ AssetHandle<Host::RenderObject> FindByDAG(const std::string& id)
        {
            auto it = m_dagMap.find(id);
            return (it == m_dagMap.end()) ? AssetHandle<Host::RenderObject>(nullptr) : it->second;
        }

        template<typename T>
        __host__ AssetHandle<T> FindFirstOfType()
        {
            for (auto object : m_objectMap)
            {
                auto downcast = object.second.DynamicCast<T>();
                if (downcast)
                {
                    return downcast;
                }
            }
            return nullptr;
        }

        __host__ void Bind();
        __host__ void Finalise() const;
        __host__ virtual void Synchronise() override final;

        __host__ bool Exists(const std::string& id) const { return m_objectMap.find(id) != m_objectMap.end(); }
        __host__ size_t Size() const { return m_objectMap.size(); }

        __host__ void Emplace(AssetHandle<Host::RenderObject>& newObject);
    };
}