#pragma once

#include "math/CudaMath.cuh"
#include "CudaAsset.cuh"
#include <map>

namespace Cuda
{
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
        class Tracable;
        class Light;
        class Material;
        class BxDF;
        class RenderObjectContainer;

        using TracableContainer = AssetHandle<Host::AssetContainer<Host::Tracable>>;
        using LightContainer = AssetHandle<Host::AssetContainer<Host::Light>>;
        using MaterialContainer = AssetHandle<Host::AssetContainer<Host::Material>>;
        using BxDFContainer = AssetHandle<Host::AssetContainer<Host::BxDF>>;
        
        class RenderObject : public Host::Asset
        {
        public:
            __host__ virtual void Bind(RenderObject& parentObject) {}

        protected:
            __host__ RenderObject() = default;
            __host__ virtual ~RenderObject() = default; 

            void SetRenderObjects(AssetHandle<Host::RenderObjectContainer>& renderObjects)
            {
                Assert(renderObjects);
                m_renderObjects = renderObjects;
            }

        private:
            AssetHandle<Host::RenderObjectContainer>   m_renderObjects;
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
                m_objectMap.emplace(newObject);
            }
        }
    private:
        std::map<std::string, AssetHandle<Host::RenderObject>> m_objectMap;
    };
}