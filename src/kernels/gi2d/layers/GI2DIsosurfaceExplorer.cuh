#pragma once

#include "UILayer.cuh"

#include "../BIH2DAsset.cuh"
#include "../tracables/Tracable.cuh"
#include "../Transform2D.cuh"

using namespace Cuda;

namespace GI2D
{
    class UIInspector;
    
    struct IsosurfaceExplorerParams
    {
        __host__ __device__ IsosurfaceExplorerParams();

        struct
        {
            uint width;
            uint height;
            int downsample;
        }
        m_accum;

        bool m_isDirty;
        int m_frameIdx;
    };

    struct IsosurfaceExplorerObjects
    {
        VectorInterface<GI2D::TracableInterface*>* m_inspectors = nullptr;
        Cuda::Device::ImageRGBW* m_accumBuffer = nullptr;
    };

    namespace Device
    {
        class IsosurfaceExplorer : public Cuda::Device::Asset,
            public UILayerParams,
            public IsosurfaceExplorerParams,
            public IsosurfaceExplorerObjects
        {
        public:
            __host__ __device__ IsosurfaceExplorer();

            __device__ void Render();
            __device__ void Composite(Cuda::Device::ImageRGBA* outputImage);
        };
    }

    namespace Host
    {
        class IsosurfaceExplorer : public UILayer,
                                   public IsosurfaceExplorerParams
        {
        public:
            IsosurfaceExplorer(const std::string& id, AssetHandle<Host::BIH2DAsset>& bih, AssetHandle<TracableContainer>& tracables, AssetHandle<InspectorContainer>& inspectors,
                               const uint width, const uint height, const uint downsample, cudaStream_t renderStream);
            virtual ~IsosurfaceExplorer();

            __host__ virtual void Render() override final;
            __host__ virtual void Composite(AssetHandle<Cuda::Host::ImageRGBA>& hostOutputImage) const override final;
            __host__ void OnDestroyAsset();

            __host__ void Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx);

        protected:
            __host__ void Synchronise(const int syncType);

        private:
            GI2D::Device::IsosurfaceExplorer* cu_deviceData = nullptr;
            IsosurfaceExplorerObjects         m_deviceObjects;

            AssetHandle<Cuda::Host::ImageRGBW>  m_hostAccumBuffer;
            AssetHandle<InspectorContainer>     m_hostInspectors;
        };
    }
}