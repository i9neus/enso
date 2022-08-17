#pragma once

#include "../CudaAsset.cuh"
#include "../CudaManagedObject.cuh"
#include "../CudaImage.cuh"
#include "../CudaVector.cuh"

#include "CudaBIH2D.cuh"
#include "CudaPrimitive2D.cuh"

using namespace Cuda;

namespace GI2D
{         
    struct OverlayParams
    {
        __host__ __device__ OverlayParams();

        ViewTransform view;
        BBox2f sceneBounds;

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
                Cuda::Device::Vector<LineSegment>* lineSegments = nullptr;
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
        class Overlay : public Cuda::Host::Asset
        {
        public:
            Overlay(const std::string& id, AssetHandle<Host::BIH2DAsset>& bih, AssetHandle<Cuda::Host::Vector<LineSegment>>& lineSegments,
                        const uint width, const uint height, cudaStream_t renderStream);
            virtual ~Overlay();

            __host__ void Render();
            __host__ void Composite(AssetHandle<Cuda::Host::ImageRGBA>& hostOutputImage);
            __host__ void OnDestroyAsset();
            __host__ void SetParams(const OverlayParams& newParams);

        private:
            OverlayParams               m_params;
            Device::Overlay::Objects    m_objects;
            Device::Overlay*            cu_deviceData = nullptr;

            AssetHandle<Host::BIH2DAsset>               m_hostBIH2D;
            AssetHandle<Cuda::Host::Vector<LineSegment>>      m_hostLineSegments;

            AssetHandle<Cuda::Host::ImageRGBW>          m_hostAccumBuffer;
        };
    }
}