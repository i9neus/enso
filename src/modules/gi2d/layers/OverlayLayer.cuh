#pragma once

#include "UILayer.cuh"

#include "../bih/BIH2DAsset.cuh"
//#include "../tracables/primitives/Primitive2D.cuh"
#include "../SceneObject.cuh"

namespace Enso
{         
    namespace Device { class Tracable; }
    
    struct OverlayLayerParams
    {
        __host__ __device__ OverlayLayerParams();
        __device__ void Validate() const {}

        UIGridCtx           gridCtx;
        UIViewCtx           viewCtx;
        
        struct
        {
            BBox2f                  mouseBBox;
            BBox2f                  lassoBBox;
            BBox2f                  selectedBBox;
            int                     numSelected;
            bool                    isLassoing = false;
        } selectionCtx;
    };

    struct OverlayLayerObjects
    {
        __device__ void Validate() const
        {
            CudaAssert(scene);
            CudaAssert(accumBuffer);
        }
        
        const Device::SceneContainer*     scene = nullptr;
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

            __device__ void Synchronise(const OverlayLayerParams& params) { m_params = params; }
            __device__ void Synchronise(const OverlayLayerObjects& objects) { objects.Validate(); m_objects = objects; }

        private:
            OverlayLayerParams              m_params;
            OverlayLayerObjects             m_objects;

            Device::SceneContainer        m_scene;
        };
    }

    namespace Host
    {
        class OverlayLayer : public Host::UILayer,
                             public Host::GenericObject
        {
        public:
            OverlayLayer(const Asset::InitCtx& initCtx, const AssetHandle<const Host::SceneContainer>& scene, const uint width, const uint height, cudaStream_t renderStream);
            virtual ~OverlayLayer() noexcept;

            __host__ virtual void Render() override final;
            __host__ virtual void Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const override final; 

            __host__ virtual bool Prepare(const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx);

            __host__ static const std::string  GetAssetClassStatic() { return "overlaylayer"; }
            __host__ virtual std::string       GetAssetClass() const override final { return GetAssetClassStatic(); }

        protected:
            __host__ void Synchronise(const int syncType);

        private:
            const AssetHandle<const Host::SceneContainer>& m_scene;

            Device::OverlayLayer*               cu_deviceInstance = nullptr;
            Device::OverlayLayer                m_hostInstance;
            OverlayLayerObjects                 m_deviceObjects;
            OverlayLayerParams                  m_params;

            AssetHandle<Host::ImageRGBW>        m_hostAccumBuffer;
        };
    }
}