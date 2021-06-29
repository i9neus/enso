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
            RenderObject() = default;

        protected:
            __device__ ~RenderObject() = default;
        };
    }

    namespace Host
    {        
        class RenderObject : public Host::Asset
        {
        public:
            __host__ virtual void Bind(RenderObjectContainer& objectContainer) {}

        protected:
            __host__ RenderObject() = default;
            __host__ virtual ~RenderObject() = default; 

            template<typename ThisType, typename BindType>
            __host__ AssetHandle<BindType> GetAssetHandleForBinding(RenderObjectContainer& objectContainer, const std::string& otherId)
            {
                // Try to find a handle to the asset
                AssetHandle<RenderObject> baseAsset = objectContainer.Find(otherId);
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
        };     
    }  

    class RenderObjectContainer : public Host::Asset
    {
    public:
        using ObjectMap = std::map<std::string, AssetHandle<Host::RenderObject>>;

        class Iterator
        {
        private:
            ObjectMap::iterator m_it;
        public:
            Iterator(ObjectMap::iterator it) noexcept : m_it(it) {}

            Iterator& operator++() { ++m_it; return *this; }
            bool operator!=(const Iterator& rhs) const { return m_it != rhs.m_it; }
            AssetHandle<Host::RenderObject>& operator*() { return m_it->second; }
        };

        __host__ RenderObjectContainer() = default;

        __host__ virtual void OnDestroyAsset() override final
        {
            for (auto& object : m_objectMap)
            {
                object.second.DestroyAsset();
            }
        }

        __host__ Iterator begin() noexcept { return Iterator(m_objectMap.begin()); }
        __host__ Iterator end() noexcept { return Iterator(m_objectMap.end()); }

        __host__ void Bind()
        {
            for (auto& object : m_objectMap)
            {
                object.second->Bind(*this);
            }
        }

        __host__ AssetHandle<Host::RenderObject> Find(const std::string& id)
        {
            auto it = m_objectMap.find(id);
            return (it == m_objectMap.end()) ? AssetHandle<Host::RenderObject>(nullptr) : it->second;
        }

        __host__ virtual void Synchronise() override final
        {
            for (auto& object : m_objectMap)
            {
                object.second->Synchronise();
            }
        }

        __host__ bool Exists(const std::string& id) const { return m_objectMap.find(id) != m_objectMap.end(); }
        __host__ size_t Size() const { return m_objectMap.size(); }

        __host__ void Emplace(AssetHandle<Host::RenderObject>& newObject)
        {
            AssertMsgFmt(!Exists(newObject->GetAssetID()), "A render object with ID '%s' already exists in the object container.\n", newObject->GetAssetID().c_str());

            m_objectMap[newObject->GetAssetID()] = newObject;
        }

    private:
        ObjectMap m_objectMap;
    };
}