#pragma once

#include "math/CudaMath.cuh"
#include "CudaAsset.cuh"
#include <map>

namespace Cuda
{
    class RenderObjectContainer;

    enum class RenderObjectContainerResult : uint { kSuccess = 0, kNotFound, kInvalidType };
    enum LightIdFlags : uchar { kNotALight = 0xff };
    
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
            __host__ virtual std::vector<AssetHandle<Host::RenderObject>> GetChildObjectHandles() { return std::vector<AssetHandle<Host::RenderObject>>();  }
            __host__ void Host::RenderObject::UpdateDAGPath(const ::Json::Node& node);

            __host__ const std::string& GetDAGPath() const { return m_dagPath; }
            __host__ const bool HasDAGPath() const { return !m_dagPath.empty(); }
            __host__ bool IsChildObject() const { return m_isChildObject; }

            __host__ virtual void OnPreRender() {}
            __host__ virtual void OnPostRender() {}
            __host__ virtual void OnPreRenderPass(const float wallTime, const float frameIdx) {}
            __host__ virtual void OnPostRenderPass() {}

        protected:
            __host__ RenderObject() : m_isChildObject(false) {}
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
            __host__ void MakeChildObject(const bool isChild = true) { m_isChildObject = isChild; }

        private:
            std::string         m_dagPath;
            bool                m_isChildObject;
        };     
    }  

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
        __host__ AssetHandle<ObjectType> FindByID(const std::string& id)
        {
            auto it = m_objectMap.find(id);
            return (it == m_objectMap.end()) ? AssetHandle<ObjectType>(nullptr) : it->second.DynamicCast<ObjectType>();
        }

        __host__ AssetHandle<Host::RenderObject> FindByDAG(const std::string& id)
        {
            auto it = m_dagMap.find(id);
            return (it == m_dagMap.end()) ? AssetHandle<Host::RenderObject>(nullptr) : it->second;
        }        

        template<typename T>
        __host__ std::vector<AssetHandle<T>> FindAllOfType(std::function<bool(const AssetHandle<T>&)> comparator = nullptr, const bool findFirst = false)
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
        __host__ AssetHandle<T> FindFirstOfType(std::function<bool(const AssetHandle<T>&)> comparator = nullptr)
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