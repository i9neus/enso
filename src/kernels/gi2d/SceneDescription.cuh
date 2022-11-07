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
            
            Core::Vector<Device::Tracable*>*    tracables = nullptr;
            BIH2D<BIH2DFullNode>*               bih = nullptr;

            Device::VoxelProxyGrid*             voxelProxy;
        };
    }

    namespace Host
    {
        using TracableContainer = ::Core::Host::AssetVector<Host::TracableInterface, Device::Tracable>;
        using InspectorContainer = ::Core::Host::AssetVector<Host::TracableInterface, Device::Tracable>;
        
        class SceneDescription : public Cuda::Host::AssetAllocator
        {
        public:
            __host__                SceneDescription(const std::string& id);
            __host__ virtual        ~SceneDescription();
            __host__ virtual void   OnDestroyAsset() override final {}
            
            __host__ void           Prepare();
            __host__ const Device::SceneDescription& GetDeviceObjects() const { return m_deviceObjects; }

            // Geometry
            AssetHandle<Host::BIH2DAsset>           sceneBIH;
            AssetHandle<TracableContainer>          hostTracables;
            AssetHandle<InspectorContainer>         hostInspectors;
            
            // Voxel grids
            AssetHandle<Host::VoxelProxyGrid>       voxelProxy;

        private:
            Device::SceneDescription                m_deviceObjects;
        };
    };
}