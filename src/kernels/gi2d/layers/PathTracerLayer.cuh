#pragma once

#include "UILayer.cuh"
#include "../integrators/PathTracer2D.cuh"
#include "../integrators/Camera2D.cuh"
#include "../SceneDescription.cuh"

using namespace Cuda;

namespace GI2D
{     
    struct PathTracerLayerParams
    {
        __host__ __device__ PathTracerLayerParams();

        struct
        {
            int downsample;
        }
        m_accum;

        bool m_isDirty;
        int m_frameIdx;        
    };

    struct PathTracerLayerObjects
    {
        Device::SceneDescription                        m_scene;
        Cuda::Device::ImageRGBW*                        m_accumBuffer = nullptr;
    };

    namespace Device
    {
        class PathTracerLayer : public UILayer,
                                public PathTracerLayerParams,
                                public PathTracerLayerObjects,
                                public Camera2D
        {
        public:
            __device__ PathTracerLayer();

            __device__ void Render();
            __device__ void Composite(Cuda::Device::ImageRGBA* outputImage);

            __device__ virtual bool CreateRay(Ray2D& ray, RenderCtx& renderCtx) const override final;
            __device__ virtual void Accumulate(const vec4& L, const RenderCtx& ctx) override final;

            __device__ virtual void OnSynchronise(const int) override final;


        private:
            PathTracer2D                            m_overlayTracer;
            //PathTracer2D                            m_voxelTracer;
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
            __host__ virtual void Composite(AssetHandle<Cuda::Host::ImageRGBA>& hostOutputImage) const override final;
            __host__ void OnDestroyAsset();

            __host__ void Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx);

        protected:
            __host__ void Synchronise(const int syncType);

        private:
            GI2D::Device::PathTracerLayer*          cu_deviceData = nullptr;
            PathTracerLayerObjects                  m_deviceObjects;

            AssetHandle<Cuda::Host::ImageRGBW>      m_hostAccumBuffer;      
        };
    }
}