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
            __device__ void OnSynchronise(const uint) {}
            __device__ void Synchronise(const SceneContainer& objects) { *this = objects; }
            
            __device__ void Validate() const
            {
                CudaAssert(tracables);
                CudaAssert(lights);
                CudaAssert(sceneObjects);

                CudaAssert(tracableBIH);
                CudaAssert(sceneBIH);
            }

            const Vector<Device::Tracable*>* tracables = nullptr;
            const Vector<Device::Light*>* lights = nullptr;
            const Vector<Device::SceneObject*>* sceneObjects = nullptr;

            const BIH2D<BIH2DFullNode>* tracableBIH = nullptr;
            const BIH2D<BIH2DFullNode>* sceneBIH = nullptr;
            
            //const Device::VoxelProxyGrid*             voxelProxy = nullptr;
        };
    }

    namespace Host
    {
        class SceneBuilder;
        
        using TracableContainer = Host::AssetVector<Host::Tracable, Device::Tracable>;
        using LightContainer = Host::AssetVector<Host::Light, Device::Light>;
        using CameraContainer = Host::AssetVector<Host::Camera, Device::Camera>;
        using SceneObjectContainer = Host::AssetVector<Host::SceneObject, Device::SceneObject>;

        class SceneContainer : public Host::GenericObject
        {
            friend class SceneBuilder;

        public:
            __host__                SceneContainer(const Asset::InitCtx& initCtx);
            __host__ virtual        ~SceneContainer() noexcept;

            __host__ void Clear();
            __host__ void Destroy();
            __host__ const Device::SceneContainer* GetDeviceInstance() const { return cu_deviceInstance; }

            __host__ Host::BIH2DAsset& TracableBIH() { DAssert(m_hostTracableBIH);  return *m_hostTracableBIH; }
            __host__ Host::BIH2DAsset& SceneBIH() { DAssert(m_hostSceneBIH);  return *m_hostSceneBIH; }
            __host__ CameraContainer& Cameras() { DAssert(m_hostCameras); return *m_hostCameras; }

            __host__ TracableContainer& Tracables() { DAssert(m_hostTracableBIH); return *m_hostTracables; }
            __host__ TracableContainer& Lights() { DAssert(m_hostTracableBIH); return *m_hostTracables; }

            __host__ SceneObjectContainer& SceneObjects() { return *m_hostSceneObjects; }
            __host__ Host::GenericObjectContainer& GenericObjects() { return *m_hostGenericObjects; }

            __host__ virtual void Synchronise(const uint flags = 0u) override final;
            __host__ virtual bool Serialise(Json::Node& rootNode, const int flags) const override final;

            __host__ void Summarise() const;

            __host__ void Emplace(AssetHandle<Host::GenericObject>& newObject);

        private:    
            // Geometry
            AssetHandle<Host::BIH2DAsset>           m_hostTracableBIH;
            AssetHandle<Host::BIH2DAsset>           m_hostSceneBIH;

            AssetHandle<TracableContainer>          m_hostTracables;
            AssetHandle<LightContainer>             m_hostLights;
            AssetHandle<CameraContainer>            m_hostCameras;
            AssetHandle<SceneObjectContainer>       m_hostSceneObjects;

            AssetHandle<Host::GenericObjectContainer> m_hostGenericObjects;

            Device::SceneContainer*                 cu_deviceInstance = nullptr;
            Device::SceneContainer                  m_deviceObjects;

            AssetAllocator                          m_allocator;
        };
    }
}