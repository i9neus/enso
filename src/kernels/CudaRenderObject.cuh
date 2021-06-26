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
        __host__ RenderObjectContainer() = default;

        bool Exists(const std::string& id) const { return m_objectMap.find(id) != m_objectMap.end(); }

        void Emplace(AssetHandle<Host::RenderObject>& newObject)
        {
            if (!Exists(newObject->GetAssetID()))
            {
                m_objectMap[newObject->GetAssetID()] = newObject;
            }
        }
    private:
        std::map<std::string, AssetHandle<Host::RenderObject>> m_objectMap;
    };
}