#pragma once

#include "UILayer.cuh"
#include "../integrators/PathTracer2D.cuh"
#include "../integrators/Camera2D.cuh"
#include "../SceneDescription.cuh"

namespace Enso
{     
    struct PathTracerLayerParams
    {
        __host__ __device__ PathTracerLayerParams();

        struct
        {
            int downsample;
        }
        accum;

        bool m_isDirty;   
    };

    struct PathTracerLayerObjects
    {
        const Device::SceneDescription*           scenePtr = nullptr;
        Device::ImageRGBW*                        accumBuffer = nullptr;
    };

    namespace Device
    {
        class PathTracerLayer : public UILayer,
                                public Camera2D
        {
        public:
            __device__ PathTracerLayer() : m_overlayTracer(m_scene) {}

            __device__ void                     Prepare(const uint dirtyFlags);
            __device__ void                     Render();
            __device__ void                     Composite(Device::ImageRGBA* outputImage);

            __device__ virtual bool             CreateRay(Ray2D& ray, HitCtx2D& hit, RenderCtx& renderCtx) const override final;
            __device__ virtual void             Accumulate(const vec4& L, const RenderCtx& ctx) override final;

            __host__ __device__ virtual void    OnSynchronise(const int) override final;
            __device__ void                     Synchronise(const PathTracerLayerParams& params) { m_params = params; }
            __device__ void                     Synchronise(const PathTracerLayerObjects& objects) { m_objects = objects; }

        private:
            PathTracer2D                            m_overlayTracer;
            int                                     m_frameIdx;

            Device::SceneDescription                m_scene;

            PathTracerLayerParams                   m_params;
            PathTracerLayerObjects                  m_objects;
        };
    }

    namespace Host
    {
        class PathTracerLayer : public UILayer
        {
        public:
            PathTracerLayer(const std::string& id, const AssetHandle<Host::SceneDescription>& scene, const uint width, const uint height, const uint downsample, cudaStream_t renderStream);
            virtual ~PathTracerLayer();
           
            __host__ virtual void Render() override final;
            __host__ virtual void Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const override final;
            __host__ void OnDestroyAsset();

            __host__ void Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx);

        protected:
            __host__ void Synchronise(const int syncType);

        private:
            Device::PathTracerLayer*          cu_deviceData = nullptr;
            PathTracerLayerObjects            m_deviceObjects;
            PathTracerLayerParams             m_params;

            AssetHandle<Host::ImageRGBW>      m_hostAccumBuffer;
        };
    }
}