#pragma once

#include "FwdDecl.cuh"
#include "tracables/Tracable.cuh"

namespace GI2D
{
    namespace Device
    {
        struct SceneDescription : public Cuda::Device::Asset
        {
            __device__ SceneDescription() {}
            __device__ void OnSynchronise(const uint) {}
          
            const Core::Vector<Device::Tracable*>*    tracables = nullptr;
            const Core::Vector<Device::Light*>*       lights = nullptr;
            const BIH2D<BIH2DFullNode>*               tracableBIH = nullptr;

            //const Device::VoxelProxyGrid*             voxelProxy = nullptr;
        };
    }

    namespace Host
    {
        using TracableContainer = ::Core::Host::AssetVector<Host::Tracable, Device::Tracable>;
        using LightContainer = ::Core::Host::AssetVector<Host::Light, Device::Light>;
        
        class SceneDescription : public Cuda::Host::AssetAllocator
        {
        public:
            __host__                SceneDescription(const std::string& id);
            __host__ virtual        ~SceneDescription();
            __host__ virtual void   OnDestroyAsset() override final;

            __host__ void           Rebuild(Cuda::AssetHandle<Cuda::RenderObjectContainer>& renderObjects, const GI2D::UIViewCtx& viewCtx, const uint dirtyFlags);
            __host__ const Device::SceneDescription* GetDeviceInstance() const { return cu_deviceInstance; }

            __host__ Host::BIH2DAsset& TracableBIH() { DAssert(m_hostTracableBIH);  return *m_hostTracableBIH; }
            __host__ TracableContainer& Tracables() { DAssert(m_hostTracableBIH); return *m_hostTracables; }
            __host__ TracableContainer& Lights() { DAssert(m_hostTracableBIH); return *m_hostTracables; }

        private:
            // Geometry
            AssetHandle<Host::BIH2DAsset>           m_hostTracableBIH;
            AssetHandle<TracableContainer>          m_hostTracables;
            AssetHandle<LightContainer>             m_hostLights;

            // Voxel grids
            //AssetHandle<Host::VoxelProxyGrid>       voxelProxy;
            
            Device::SceneDescription*               cu_deviceInstance = nullptr;
            Device::SceneDescription                m_deviceObjects;
        };
    };
}