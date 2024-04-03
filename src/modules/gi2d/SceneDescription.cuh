#pragma once

#include "FwdDecl.cuh"
#include "tracables/Tracable.cuh"
#include "integrators/Camera2D.cuh"

namespace Enso
{
    namespace Device
    {
        struct SceneDescription : public Device::Asset
        {
            __device__ SceneDescription() {}
            __device__ void OnSynchronise(const uint) {}

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
        using CameraContainer = Host::AssetVector<Host::Camera2D, Device::Camera2D>;
        using SceneObjectContainer = Host::AssetVector<Host::SceneObject, Device::SceneObject>;

        class SceneDescription : public Host::AssetAllocator
        {
        public:
            __host__                SceneDescription(const std::string& id);
            __host__ virtual        ~SceneDescription();
            __host__ virtual void   OnDestroyAsset() override final;

            __host__ void           Rebuild(AssetHandle<GenericObjectContainer>& renderObjects, const UIViewCtx& viewCtx, const uint dirtyFlags);
            __host__ const Device::SceneDescription* GetDeviceInstance() const { return cu_deviceInstance; }

            __host__ Host::BIH2DAsset& TracableBIH() { DAssert(m_hostTracableBIH);  return *m_hostTracableBIH; }
            __host__ Host::BIH2DAsset& SceneBIH() { DAssert(m_hostSceneBIH);  return *m_hostSceneBIH; }

            __host__ TracableContainer& Tracables() { DAssert(m_hostTracableBIH); return *m_hostTracables; }
            __host__ TracableContainer& Lights() { DAssert(m_hostTracableBIH); return *m_hostTracables; }
            __host__ SceneObjectContainer& SceneObjects() { return *m_hostSceneObjects; }

        private:    
            // Geometry
            AssetHandle<Host::BIH2DAsset>           m_hostTracableBIH;
            AssetHandle<Host::BIH2DAsset>           m_hostSceneBIH;

            AssetHandle<TracableContainer>          m_hostTracables;
            AssetHandle<LightContainer>             m_hostLights;
            //AssetHandle<CameraContainer>            m_hostCameras;
            AssetHandle<SceneObjectContainer>       m_hostSceneObjects;

            Device::SceneDescription*               cu_deviceInstance = nullptr;
            Device::SceneDescription                m_deviceObjects;
        };
    };
}