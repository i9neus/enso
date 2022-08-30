#pragma once

#include "UILayer.cuh"

#include "../BIH2DAsset.cuh"
#include "../tracables/CudaPrimitive2D.cuh"
#include "../tracables/Curve.cuh"

using namespace Cuda;

namespace Cuda
{
    //namespace Host { template<typename T> class AssetContainer; }
}

namespace GI2D
{         
    class TracableInterface;
    
    struct OverlayParams
    {
        __host__ __device__ OverlayParams();

        UIViewCtx           viewCtx;
        UIGridCtx           gridCtx;
        UISelectionCtx      selectionCtx;

        RayBasic2D          ray;
        vec2                hitPoint;
        bool                isHit;
    };

    namespace Device
    {
        class Overlay : public Cuda::Device::Asset
        {
        public:
            struct Objects
            {
                Cuda::Device::Vector<TracableInterface*>* tracables = nullptr;
                Device::BIH2DAsset* bih = nullptr;
                Cuda::Device::ImageRGBW* accumBuffer = nullptr;
            };

        public:
            __host__ __device__ Overlay(const OverlayParams& params, const Objects& objects);
            
            __device__ void Synchronise(const OverlayParams& params);
            __device__ void Synchronise(const Objects& params);
            __device__ void Render();
            __device__ void Composite(Cuda::Device::ImageRGBA* outputImage);

        private:
            OverlayParams   m_params;
            Objects         m_objects;
        };
    }

    namespace Host
    {
        class Overlay : public UILayer
        {
        public:
            Overlay(const std::string& id, AssetHandle<Host::BIH2DAsset>& bih, AssetHandle<Cuda::Host::AssetVector<Host::Tracable>>& tracables,
                    const uint width, const uint height, cudaStream_t renderStream);

            virtual ~Overlay();

            __host__ virtual void Render() override final;
            __host__ virtual void Composite(AssetHandle<Cuda::Host::ImageRGBA>& hostOutputImage) const override final; 

            __host__ void OnDestroyAsset();

        protected:
            __host__ virtual void RebuildImpl() override final;

        private:
            __host__ void TraceRay();

            OverlayParams                               m_params;
            Device::Overlay::Objects                    m_objects;
            Device::Overlay*                            cu_deviceData = nullptr;

            AssetHandle<Cuda::Host::ImageRGBW>          m_hostAccumBuffer;
        };
    }
}