#pragma once

#include "UILayer.cuh"

#include "../bih/BIH2DAsset.cuh"
//#include "../tracables/primitives/Primitive2D.cuh"
#include "../tracables/LineStrip.cuh"

namespace Enso
{         
    namespace Device { class Tracable; }
    
    struct OverlayLayerParams
    {
        __host__ __device__ OverlayLayerParams();
        __device__ void Validate() const {}

        UIGridCtx           gridCtx;
        UIViewCtx           viewCtx;
        UISelectionCtx      selectionCtx;
    };

    struct OverlayLayerObjects
    {
        __device__ void Validate() const
        {
            assert(scene);
            assert(accumBuffer);
        }
        
        const Device::SceneDescription*     scene = nullptr;
        Device::ImageRGBW*                  accumBuffer = nullptr;
    };

    namespace Device
    {
        class OverlayLayer : public Device::GenericObject
        {
        public:
            __host__ __device__ OverlayLayer() {}
            
            __device__ void Prepare(const uint dirtyFlags);
            __device__ void Render();
            __device__ void Composite(Device::ImageRGBA* outputImage);

            __host__ __device__ virtual void OnSynchronise(const int) override final;
            __device__ void Synchronise(const OverlayLayerParams& params) { m_params = params; }
            __device__ void Synchronise(const OverlayLayerObjects& objects) { objects.Validate(); m_objects = objects; }

        private:
            OverlayLayerParams              m_params;
            OverlayLayerObjects             m_objects;

            Device::SceneDescription        m_scene;
        };
    }

    namespace Host
    {
        class OverlayLayer : public Host::UILayer,
                             public Host::GenericObject
        {
        public:
            OverlayLayer(const std::string& id, const AssetHandle<const Host::SceneDescription>& scene, const uint width, const uint height, cudaStream_t renderStream);

            virtual ~OverlayLayer();

            __host__ virtual void Render() override final;
            __host__ virtual void Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const override final; 

            __host__ virtual void Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx) override final;

            __host__ void OnDestroyAsset();

        protected:
            __host__ void Synchronise(const int syncType);

        private:
            const AssetHandle<const Host::SceneDescription>& m_scene;

            Device::OverlayLayer*               cu_deviceInstance = nullptr;
            Device::OverlayLayer                m_hostInstance;
            OverlayLayerObjects                 m_deviceObjects;
            OverlayLayerParams                  m_params;

            AssetHandle<Host::ImageRGBW>        m_hostAccumBuffer;
        };
    }
}