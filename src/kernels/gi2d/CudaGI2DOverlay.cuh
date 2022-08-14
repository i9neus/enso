#pragma once

#include "../CudaAsset.cuh"
#include "../CudaManagedObject.cuh"
#include "../CudaImage.cuh"
#include "../CudaVector.cuh"

#include "CudaBIH2D.cuh"
#include "CudaPrimitive2D.cuh"

namespace Cuda
{     
    enum GI2DOverlayFlags : int { kInvalidSegment = -1 };
    
    struct GI2DOverlayParams
    {
        __host__ __device__ GI2DOverlayParams();

        mat3 viewMatrix;        
        BBox2f sceneBounds;
        float viewScale;
        float dPdXY;

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
            BBox2f bBox;
            bool isLassoing;
            uint numSelected;
        }
        selection;
    };

    namespace Device
    {
        class GI2DOverlay : public Device::Asset
        {
        public:
            struct Objects
            {
                Device::Vector<LineSegment>* lineSegments = nullptr;
                Device::BIH2DAsset* bih = nullptr;
            };

        public:
            __host__ __device__ GI2DOverlay(const GI2DOverlayParams& params, const Objects& objects);
            
            __device__ void Synchronise(const GI2DOverlayParams& params);
            __device__ void Synchronise(const Objects& params);
            __device__ void Render(Device::ImageRGBA* outputImage);

        private:
            GI2DOverlayParams   m_params;
            Objects             m_objects;
        };
    }

    namespace Host
    {
        class GI2DOverlay : public Host::Asset
        {
        public:
            GI2DOverlay(const std::string& id, AssetHandle<Host::BIH2DAsset>& bih, AssetHandle<Host::Vector<LineSegment>>& lineSegments);
            virtual ~GI2DOverlay();

            __host__ void Render(AssetHandle<Host::ImageRGBA>& hostOutputImage);
            __host__ void OnDestroyAsset();
            __host__ void SetParams(const GI2DOverlayParams& newParams);

        private:
            GI2DOverlayParams               m_params;
            Device::GI2DOverlay::Objects    m_objects;
            Device::GI2DOverlay*            cu_deviceData = nullptr;

            AssetHandle<Host::BIH2DAsset>               m_hostBIH2D;
            AssetHandle<Host::Vector<LineSegment>>      m_hostLineSegments;
        };
    }
}