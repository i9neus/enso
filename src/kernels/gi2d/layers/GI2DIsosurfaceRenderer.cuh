#pragma once

#include "UILayer.cuh"

#include "../BIH2DAsset.cuh"

using namespace Cuda;

namespace Cuda
{
    //namespace Host { template<typename T> class AssetContainer; }
}

namespace GI2D
{
    class TracableInterface;

    struct IsosurfaceRendererParams
    {
        __host__ __device__ IsosurfaceRendererParams();

        UIViewCtx           viewCtx;
        UISelectionCtx      selectionCtx;
    };

    namespace Device
    {
        class IsosurfaceRenderer : public Cuda::Device::Asset
        {
        public:
            struct Objects
            {
                Cuda::Device::Vector<TracableInterface*>* tracables = nullptr;
                Device::BIH2DAsset* bih = nullptr;
                Cuda::Device::ImageRGBW* accumBuffer = nullptr;
            };

        public:
            __host__ __device__ IsosurfaceRenderer(const IsosurfaceRendererParams& params, const Objects& objects);

            __device__ void Synchronise(const IsosurfaceRendererParams& params);
            __device__ void Synchronise(const Objects& params);
            __device__ void Render();
            __device__ void Composite(Cuda::Device::ImageRGBA* outputImage);

        private:
            IsosurfaceRendererParams   m_params;
            Objects         m_objects;
        };
    }

    namespace Host
    {
        class IsosurfaceRenderer : public UILayer
        {
        public:
            IsosurfaceRenderer(const std::string& id, AssetHandle<Host::BIH2DAsset>& bih, AssetHandle<TracableContainer>& tracables,
                const uint width, const uint height, cudaStream_t renderStream);

            virtual ~IsosurfaceRenderer();

            __host__ virtual void Render() override final;
            __host__ virtual void Composite(AssetHandle<Cuda::Host::ImageRGBA>& hostOutputImage) const override final;
            __host__ void OnDestroyAsset();

        protected:
            __host__ virtual void RebuildImpl() override final;

        private:
            IsosurfaceRendererParams                               m_params;
            Device::IsosurfaceRenderer::Objects                    m_objects;
            Device::IsosurfaceRenderer* cu_deviceData = nullptr;

            AssetHandle<Cuda::Host::ImageRGBW>          m_hostAccumBuffer;
        };
    }
}