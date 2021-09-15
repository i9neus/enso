#pragma once

#include "math/CudaMath.cuh"
#include "CudaAsset.cuh"
#include <map>
#include "CudaRenderObject.cuh"

namespace Cuda
{    
    enum class RenderObjectContainerResult : uint { kSuccess = 0, kNotFound, kInvalidType };
    
    class RenderObjectContainer : public Host::Asset
    {
        using RenderObjectMap = std::map<std::string, AssetHandle<Host::RenderObject>>;

    private:
        RenderObjectMap       m_objectMap;
        RenderObjectMap       m_dagMap;

    public:
        template<typename ItType, bool IsConst>
        class __Iterator
        {
        private:
            ItType m_it;
        public:
            __Iterator(ItType it) noexcept : m_it(it) {}

            __Iterator& operator++() { ++m_it; return *this; }
            bool operator!=(const __Iterator& rhs) const { return m_it != rhs.m_it; }

            template<bool C = IsConst> inline typename std::enable_if<!C, AssetHandle<Host::RenderObject>&>::type operator*() { return m_it->second; }
            template<bool C = IsConst> inline typename std::enable_if<C, const AssetHandle<Host::RenderObject>&>::type operator*() const { return m_it->second; }
        };

        using Iterator = __Iterator<RenderObjectMap::iterator, false>;
        using ConstIterator = __Iterator<RenderObjectMap::const_iterator, true>;

        __host__ RenderObjectContainer() = default;
        __host__ RenderObjectContainer(const RenderObjectContainer&) = delete;
        __host__ RenderObjectContainer(const RenderObjectContainer&&) = delete;
        __host__ virtual void OnDestroyAsset() override final;

        __host__ Iterator begin() noexcept { return Iterator(m_objectMap.begin()); }
        __host__ Iterator end() noexcept { return Iterator(m_objectMap.end()); }
        __host__ ConstIterator begin() const noexcept { return ConstIterator(m_objectMap.cbegin()); }
        __host__ ConstIterator end() const noexcept { return ConstIterator(m_objectMap.cend()); }

        template<typename ObjectType = Host::RenderObject>
        __host__ AssetHandle<ObjectType> FindByID(const std::string& id) const
        {
            auto it = m_objectMap.find(id);
            return (it == m_objectMap.end()) ? AssetHandle<ObjectType>(nullptr) : it->second.DynamicCast<ObjectType>();
        }

        __host__ AssetHandle<Host::RenderObject> FindByDAG(const std::string& id) const
        {
            auto it = m_dagMap.find(id);
            return (it == m_dagMap.end()) ? AssetHandle<Host::RenderObject>(nullptr) : it->second;
        }

        template<typename T>
        __host__ std::vector<AssetHandle<T>> FindAllOfType(std::function<bool(const AssetHandle<T>&)> comparator = nullptr, const bool findFirst = false) const
        {
            std::vector<AssetHandle<T>> assets;
            for (auto object : m_objectMap)
            {
                auto downcast = object.second.DynamicCast<T>();
                if (downcast)
                {
                    if (!comparator || comparator(downcast))
                    {
                        assets.push_back(downcast);
                        if (findFirst) { break; }
                    }
                }
            }
            return assets;
        }

        template<typename T>
        __host__ AssetHandle<T> FindFirstOfType(std::function<bool(const AssetHandle<T>&)> comparator = nullptr) const
        {
            auto handles = FindAllOfType<T>(comparator, true);
            return handles.empty() ? nullptr : handles.front();
        }

        __host__ void Bind();
        __host__ void Finalise() const;
        __host__ virtual void Synchronise() override final;

        __host__ bool Exists(const std::string& id) const { return m_objectMap.find(id) != m_objectMap.end(); }
        __host__ size_t Size() const { return m_objectMap.size(); }

        __host__ void Emplace(AssetHandle<Host::RenderObject>& newObject);
    };
}