#pragma once

#include "UILayer.cuh"

#include "../BIH2DAsset.cuh"
#include "../tracables/Tracable.cuh"
#include "../Transform2D.cuh"

using namespace Cuda;

namespace GI2D
{     
    struct PathTracerParams
    {
        __host__ __device__ PathTracerParams();

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

    struct PathTracerObjects
    {
        VectorInterface<GI2D::TracableInterface*>*      m_tracables = nullptr;
        BIH2D<BIH2DFullNode>*                           m_bih = nullptr;
        Cuda::Device::ImageRGBW*                        m_accumBuffer = nullptr;
    };

    namespace Device
    {
        class PathTracer : public Cuda::Device::Asset,
                           public UILayerParams,
                           public PathTracerParams,
                           public PathTracerObjects
        {
        public:
            __host__ __device__ PathTracer();

            __device__ void Render();
            __device__ void Composite(Cuda::Device::ImageRGBA* outputImage);

        private:

        };
    }

    namespace Host
    {
        class PathTracer : public UILayer, 
                           public PathTracerParams
        {
        public:
            PathTracer(const std::string& id, AssetHandle<Host::BIH2DAsset>& bih, AssetHandle<TracableContainer>& tracables, 
                       const uint width, const uint height, const uint downsample, cudaStream_t renderStream);
            virtual ~PathTracer();
           
            __host__ virtual void Render() override final;
            __host__ virtual void Composite(AssetHandle<Cuda::Host::ImageRGBA>& hostOutputImage) const override final;
            __host__ void OnDestroyAsset();

            __host__ void Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx);

        protected:
            __host__ void Synchronise(const int syncType);

        private:
            GI2D::Device::PathTracer*               cu_deviceData = nullptr;
            PathTracerObjects                       m_deviceObjects;

            AssetHandle<Cuda::Host::ImageRGBW>      m_hostAccumBuffer;      
        };
    }
}