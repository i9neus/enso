#pragma once

#include "UILayer.cuh"

#include "../BIH2DAsset.cuh"
//#include "../tracables/primitives/CudaPrimitive2D.cuh"
#include "../tracables/Curve.cuh"

using namespace Cuda;

namespace GI2D
{         
    namespace Device { class Tracable; }
    
    struct OverlayLayerParams
    {
        __host__ __device__ OverlayLayerParams();

        UIGridCtx           m_gridCtx;
    };

    struct OverlayLayerObjects
    {
        Device::SceneDescription                    m_scene;
        Cuda::Device::ImageRGBW*                    m_accumBuffer = nullptr;
    };

    namespace Device
    {
        class OverlayLayer : public Device::UILayer,
                             public OverlayLayerParams,
                             public OverlayLayerObjects
        {
        public:
            __host__ __device__ OverlayLayer() {}
            
            __device__ void Render();
            __device__ void Composite(Cuda::Device::ImageRGBA* outputImage);
        };
    }

    namespace Host
    {
        class OverlayLayer : public UILayer,
                        public OverlayLayerParams
        {
        public:
            OverlayLayer(const std::string& id, const AssetHandle<Host::SceneDescription>& scene, const uint width, const uint height, cudaStream_t renderStream);

            virtual ~OverlayLayer();

            __host__ virtual void Render() override final;
            __host__ virtual void Composite(AssetHandle<Cuda::Host::ImageRGBA>& hostOutputImage) const override final; 
            __host__ void Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx);

            __host__ void OnDestroyAsset();

        protected:
            __host__ void Synchronise(const int syncType);

        private:
            //__host__ void TraceRay();

            Device::OverlayLayer*                       cu_deviceData = nullptr;
            OverlayLayerObjects                         m_deviceObjects;

            AssetHandle<Cuda::Host::ImageRGBW>          m_hostAccumBuffer;
            AssetHandle<InspectorContainer>             m_hostInspectors;
        };
    }
}