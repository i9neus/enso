#pragma once

#include "UILayer.cuh"

#include "../BIH2DAsset.cuh"
//#include "../tracables/primitives/CudaPrimitive2D.cuh"
//#include "../tracables/Curve.cuh"

using namespace Cuda;

namespace GI2D
{         
    namespace Device { class Tracable; }
    
    struct OverlayParams
    {
        __host__ __device__ OverlayParams();

        UIGridCtx           m_gridCtx;
    };

    struct OverlayObjects
    {
        Core::Device::Vector<Device::Tracable*>*    m_tracables = nullptr;
        Core::Device::Vector<Device::Tracable*>*    m_inspectors = nullptr;
        BIH2D<BIH2DFullNode>*                       m_bih = nullptr;
        Cuda::Device::ImageRGBW*                    m_accumBuffer = nullptr;
    };

    namespace Device
    {
        class Overlay : public Cuda::Device::Asset,
                        public UILayerParams,
                        public OverlayParams,
                        public OverlayObjects
        {
        public:
            __host__ __device__ Overlay();
            
            __device__ void Render();
            __device__ void Composite(Cuda::Device::ImageRGBA* outputImage);
        };
    }

    namespace Host
    {
        class Overlay : public UILayer,
                        public OverlayParams
        {
        public:
            Overlay(const std::string& id, AssetHandle<Host::BIH2DAsset>& bih, AssetHandle<TracableContainer>& tracables, AssetHandle<InspectorContainer>& inspectors,
                    const uint width, const uint height, cudaStream_t renderStream);

            virtual ~Overlay();

            __host__ virtual void Render() override final;
            __host__ virtual void Composite(AssetHandle<Cuda::Host::ImageRGBA>& hostOutputImage) const override final; 
            __host__ void Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx);

            __host__ void OnDestroyAsset();

        protected:
            __host__ void Synchronise(const int syncType);

        private:
            //__host__ void TraceRay();

            Device::Overlay*                            cu_deviceData = nullptr;
            OverlayObjects                              m_deviceObjects;

            AssetHandle<Cuda::Host::ImageRGBW>          m_hostAccumBuffer;
            AssetHandle<InspectorContainer>             m_hostInspectors;
        };
    }
}