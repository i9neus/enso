#pragma once

#include "../FwdDecl.cuh"
#include "core/GenericObject.cuh"

namespace Enso
{
    namespace Device
    {
        struct SceneContainer : public Device::Asset
        {
            __device__ SceneContainer() {}
            __device__ void Synchronise(const SceneContainer& objects) { *this = objects; }
            
            __device__ void Validate() const
            {
                CudaAssert(sceneObjects);
                CudaAssert(sceneBIH);
            }

            const Vector<Device::DrawableObject*>* sceneObjects = nullptr;
            const BIH2D<BIH2DFullNode>* sceneBIH = nullptr;
        };
    }

    namespace Host
    {
        class SceneBuilder;        
    
        using DrawableObjectContainer = Host::AssetVector<Host::DrawableObject, Device::DrawableObject>;

        class SceneContainer : public Host::GenericObject
        {
            friend class SceneBuilder;

        public:
            __host__                SceneContainer(const Asset::InitCtx& initCtx);
            __host__ virtual        ~SceneContainer() noexcept;

            __host__ void           Prepare();
            __host__ void           Clean();

            __host__ void           Destroy();
            __host__ const Device::SceneContainer* GetDeviceInstance() const { return cu_deviceInstance; }

            __host__ Host::BIH2DAsset& SceneBIH() { DAssert(m_hostSceneBIH);  return *m_hostSceneBIH; }

            __host__ DrawableObjectContainer& DrawableObjects() { return *m_hostDrawableObjects; }
            __host__ Host::GenericObjectContainer& GenericObjects() { return *m_hostGenericObjects; }

            __host__ virtual void Synchronise(const uint flags = 0u) override final;
            __host__ virtual bool Serialise(Json::Node& rootNode, const int flags) const override final;

            __host__ void Summarise() const;

            __host__ void Emplace(AssetHandle<Host::GenericObject>& newObject);

        private:
            __host__ void           Clear();

        private:    
            // Geometry
            AssetHandle<Host::BIH2DAsset>           m_hostSceneBIH;

            AssetHandle<DrawableObjectContainer>       m_hostDrawableObjects;

            AssetHandle<Host::GenericObjectContainer> m_hostGenericObjects;

            Device::SceneContainer*                 cu_deviceInstance = nullptr;
            Device::SceneContainer                  m_deviceObjects;
        };
    }
}