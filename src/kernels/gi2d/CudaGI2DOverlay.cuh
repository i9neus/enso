#pragma once

#include "UILayer.cuh"

#include "BIH2DAsset.cuh"
#include "CudaPrimitive2D.cuh"
#include "Curve.cuh"

using namespace Cuda;

namespace Cuda
{
    //namespace Host { template<typename T> class AssetContainer; }
}

namespace GI2D
{         
    struct OverlayParams
    {
        __host__ __device__ OverlayParams();

        ViewTransform2D view;

        int selectedSegmentIdx;
        vec2 mousePosView;
        vec2 rayOriginView;

        struct
        {
            bool show;
            float majorLineSpacing;
            float minorLineSpacing;
            float lineAlpha;
        } 
        grid;

        struct
        {
            vec2 mouse0, mouse1;
            BBox2f lassoBBox;
            BBox2f selectedBBox;
            bool isLassoing;
            uint numSelected;
        }
        selection;
    };

    namespace Device
    {
        class Overlay : public Cuda::Device::Asset
        {
        public:
            struct Objects
            {
                Cuda::Device::Vector<Device::Tracable*>* tracables = nullptr;
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
            __host__ virtual void Synchronise() override final;

            __host__ void OnDestroyAsset();

        private:
            OverlayParams                               m_params;
            Device::Overlay::Objects                    m_objects;
            Device::Overlay*                            cu_deviceData = nullptr;

            AssetHandle<Cuda::Host::ImageRGBW>          m_hostAccumBuffer;
        };
    }
}