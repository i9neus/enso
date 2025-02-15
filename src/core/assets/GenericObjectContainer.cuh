﻿#pragma once

#include "core/math/Math.cuh"
#include "AssetAllocator.cuh"
#include <unordered_map>
#include "GenericObject.cuh"
//#include "Vector.cuh"

namespace Enso
{
    namespace Host
    {
        enum class GenericObjectContainerResult : uint { kSuccess = 0, kNotFound, kInvalidType };

        class GenericObjectContainer : public Host::Asset
        {
            // FIXME: Weak pointers need to replaced with an integrated strong/weak asset handle
            using RenderObjectMap = std::unordered_map<std::string, AssetHandle<Host::GenericObject>>;
            using WeakRenderObjectMap = std::unordered_map<std::string, WeakAssetHandle<Host::GenericObject>>;

            using WeakRenderObjectArray = std::vector< WeakAssetHandle<Host::GenericObject>>;

        private:
            RenderObjectMap                         m_objectMap;
            WeakRenderObjectMap                     m_dagMap;

            uint                                    m_uniqueIdx;

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

                template<bool C = IsConst> inline typename std::enable_if<!C, AssetHandle<Host::GenericObject>&>::type operator*() { return m_it->second; }
                template<bool C = IsConst> inline typename std::enable_if<C, const AssetHandle<Host::GenericObject>&>::type operator*() const { return m_it->second; }
            };

            using Iterator = __Iterator<RenderObjectMap::iterator, false>;
            using ConstIterator = __Iterator<RenderObjectMap::const_iterator, true>;

            __host__ GenericObjectContainer(const Asset::InitCtx& initCtx) : Asset(initCtx), m_uniqueIdx(0) {}
            __host__ GenericObjectContainer(const GenericObjectContainer&) = delete;
            __host__ GenericObjectContainer(const GenericObjectContainer&&) = delete;
            __host__ virtual ~GenericObjectContainer() noexcept;

            __host__ Iterator begin() noexcept { return Iterator(m_objectMap.begin()); }
            __host__ Iterator end() noexcept { return Iterator(m_objectMap.end()); }
            __host__ ConstIterator begin() const noexcept { return ConstIterator(m_objectMap.cbegin()); }
            __host__ ConstIterator end() const noexcept { return ConstIterator(m_objectMap.cend()); }

            __host__ bool Exists(const std::string& id) const { return m_objectMap.find(id) != m_objectMap.end(); }

            __host__ AssetHandle<Host::GenericObject> operator[](const std::string& id) const
            {
                auto it = m_objectMap.find(id);
                return (it == m_objectMap.end()) ? AssetHandle<GenericObject>(nullptr) : it->second.DynamicCast<GenericObject>();
            }

            template<typename ObjectType = Host::GenericObject>
            __host__ AssetHandle<ObjectType> FindByID(const std::string& id) const
            {
                auto it = m_objectMap.find(id);
                return (it == m_objectMap.end()) ? AssetHandle<ObjectType>(nullptr) : it->second.DynamicCast<ObjectType>();
            }

            __host__ AssetHandle<Host::GenericObject> FindByDAG(const std::string& id) const
            {
                auto it = m_dagMap.find(id);
                return (it == m_dagMap.end()) ? AssetHandle<Host::GenericObject>(nullptr) : AssetHandle<Host::GenericObject>(it->second);
            }

            template<typename DowncastType, typename OpFunctor>
            __host__ void ForEachOfType(OpFunctor functor) const
            {
                for (auto object : m_objectMap)
                {
                    auto downcast = object.second.DynamicCast<DowncastType>();
                    if (downcast)
                    {
                        if (!functor(downcast)) { return; }
                    }
                }
            }

            template<typename OpFunctor>
            __host__ void ForEach(OpFunctor functor) const
            {
                for (auto object : m_objectMap)
                {
                    if (!functor(object.second)) { return; }
                }
            }

            template<typename DowncastType>
            __host__ std::vector<AssetHandle<DowncastType>> FindAllOfType(std::function<bool(const AssetHandle<DowncastType>&)> comparator = nullptr, const bool findFirst = false) const
            {
                std::vector<AssetHandle<DowncastType>> assets;
                ForEachOfType<DowncastType>([&](AssetHandle<DowncastType>& asset) -> bool
                    {
                        if (!comparator || comparator(asset))
                        {
                            assets.push_back(asset);
                            if (findFirst) { return false; }
                        }
                return true;
                    });
                return std::move(assets);
            }

            template<typename DowncastType>
            __host__ AssetHandle<DowncastType> FindFirstOfType(std::function<bool(const AssetHandle<DowncastType>&)> comparator = nullptr) const
            {
                auto handles = FindAllOfType<DowncastType>(comparator, true);
                return handles.empty() ? nullptr : handles.front();
            }


            //__host__ void Finalise() const;
            __host__ void BindAll();
            __host__ void CleanAll();
            __host__ bool RebuildAll();
            __host__ void SynchroniseAll(const uint flags);
            __host__ uint GetUniqueIndex() { return m_uniqueIdx; }

            __host__ size_t Size() const { return m_objectMap.size(); }

            template <typename ObjectType, typename = typename std::enable_if_t<std::is_base_of<Host::GenericObject, ObjectType>::value>>
            __host__ void Emplace(AssetHandle<ObjectType>& newObject, const bool requireDAGPath = true) 
            {
                AssertMsgFmt(!Exists(newObject->GetAssetID()), "A render object with ID '%s' already exists in the object container.\n", newObject->GetAssetID().c_str());

                // Store a strong reference to the object in the object map
                m_objectMap[newObject->GetAssetID()] = newObject;
                ++m_uniqueIdx;

                // If the object has a DAG path, add it to the map alongside its weak reference
                const std::string dagPath = newObject->GetAssetDAGPath();
                if (m_dagMap.find(dagPath) == m_dagMap.end())
                {
                    m_dagMap[dagPath] = newObject.GetWeakHandle();
                }
                else
                {
                    Log::Error("Internal error: object '%s' has the same DAG path (%s) as another object.\n", newObject->GetAssetID(), dagPath);
                }
            }

            __host__ bool Erase(const Host::GenericObject& obj, const bool mustExist);
            __host__ bool Erase(const std::string& id, const bool mustExist);
            __host__ void Clear();

            template<typename DowncastType, typename DeleteFunctor>
            __host__ void Erase(DeleteFunctor& canDelete)
            {
                for (auto it = m_objectMap.begin(); it != m_objectMap.end();)
                {
                    // Call the functor to see whether it can be deleted
                    if (canDelete(it->second)) { it = m_objectMap.erase(it); }
                    else { ++it; }
                }
            }
        };
    }
}