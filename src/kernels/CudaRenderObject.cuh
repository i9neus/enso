#pragma once

#include "math/CudaMath.cuh"
#include "CudaAsset.cuh"
#include <map>

namespace Cuda
{
    class RenderObjectContainer;
    
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
            __host__ virtual void Bind(RenderObjectContainer& parentObject) {}

        protected:
            __host__ RenderObject() = default;
            __host__ virtual ~RenderObject() = default; 
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