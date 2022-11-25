#pragma once

#include "../FwdDecl.cuh"
#include "../Transform2D.cuh"
#include "../RenderCtx.cuh"

namespace Enso
{
    struct PathTracerLayerObjects
    {
        Vector<vec3>* m_tracables = nullptr;
        BIH2D<BIH2DFullNode>* m_bih = nullptr;
        Device::ImageRGBW* m_accumBuffer = nullptr;
    };

    namespace Device
    {
        class PathTracerLayer : public Device::Asset,
            public UILayerParams,
            public PathTracerLayerParams,
            public PathTracerLayerObjects
        {
        public:
            __host__ __device__ PathTracerLayer();

            __device__ void Render();
            __device__ void Composite(Device::ImageRGBA* outputImage);

        private:
            Device::PathTracer2D                            m_overlayTracer;
            Device::PathTracer2D                            m_voxelTracer;
        };
    }

    namespace Host
    {
        class PathTracerLayer : public UILayer,
            public PathTracerLayerParams
        {
        public:
            PathTracerLayer(const std::string& id, AssetHandle<Host::BIH2DAsset>& bih, AssetHandle<TracableContainer>& tracables,
                const uint width, const uint height, const uint downsample, cudaStream_t renderStream);
            virtual ~PathTracerLayer();

            __host__ virtual void Render() override final;
            __host__ virtual void Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const override final;
            __host__ void OnDestroyAsset();

            __host__ void Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx);

        protected:
            __host__ void Synchronise(const int syncType);

        private:
            Device::PathTracerLayer* cu_deviceData = nullptr;
            PathTracerLayerObjects                  m_deviceObjects;

            AssetHandle<Host::ImageRGBW>      m_hostAccumBuffer;
        };
    }
}