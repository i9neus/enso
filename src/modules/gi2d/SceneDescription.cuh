#pragma once

#include "FwdDecl.cuh"
#include "core/GenericObject.cuh"

namespace Enso
{
    namespace Device
    {
        struct SceneDescription : public Device::Asset
        {
            __device__ SceneDescription() {}
            __device__ void OnSynchronise(const uint) {}
            __device__ void Synchronise(const SceneDescription& objects) { *this = objects; }
            
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
        using TracableContainer = Host::AssetVector<Host::Tracable, Device::Tracable>;
        using LightContainer = Host::AssetVector<Host::Light, Device::Light>;
        using CameraContainer = Host::AssetVector<Host::Camera, Device::Camera>;
        using SceneObjectContainer = Host::AssetVector<Host::SceneObject, Device::SceneObject>;

        class SceneDescription : public Host::Asset
        {
        public:
            __host__                SceneDescription(const std::string& id);
            __host__ virtual        ~SceneDescription();

            __host__ void           Rebuild(AssetHandle<Host::GenericObjectContainer>& sceneObjects, const UIViewCtx& viewCtx, const uint dirtyFlags);
            __host__ const Device::SceneDescription* GetDeviceInstance() const { return cu_deviceInstance; }

            __host__ Host::BIH2DAsset& TracableBIH() { DAssert(m_hostTracableBIH);  return *m_hostTracableBIH; }
            __host__ Host::BIH2DAsset& SceneBIH() { DAssert(m_hostSceneBIH);  return *m_hostSceneBIH; }
            __host__ CameraContainer& Cameras() { DAssert(m_hostCameras); return *m_hostCameras; }

            __host__ TracableContainer& Tracables() { DAssert(m_hostTracableBIH); return *m_hostTracables; }
            __host__ TracableContainer& Lights() { DAssert(m_hostTracableBIH); return *m_hostTracables; }

            __host__ SceneObjectContainer& SceneObjects() { return *m_hostSceneObjects; }

        private:    
            // Geometry
            AssetHandle<Host::BIH2DAsset>           m_hostTracableBIH;
            AssetHandle<Host::BIH2DAsset>           m_hostSceneBIH;

            AssetHandle<TracableContainer>          m_hostTracables;
            AssetHandle<LightContainer>             m_hostLights;
            AssetHandle<CameraContainer>            m_hostCameras;
            AssetHandle<SceneObjectContainer>       m_hostSceneObjects;

            Device::SceneDescription*               cu_deviceInstance = nullptr;
            Device::SceneDescription                m_deviceObjects;

            AssetAllocator                          m_allocator;
        };
    };
}