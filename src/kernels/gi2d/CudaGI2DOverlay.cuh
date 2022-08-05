#pragma once

#include "../CudaAsset.cuh"
#include "../CudaManagedObject.cuh"
#include "../CudaImage.cuh"
#include "../CudaManagedArray.cuh"

#include "CudaBIH2D.cuh"

namespace Cuda
{     
    enum GI2DOverlayFlags : int { kInvalidSegment = -1 };

    enum GI2DLineSegmentFlags : int
    {
        kSelected = 1
    };
    
    class LineSegment
    {
    public:
        __host__ __device__ LineSegment() noexcept : flags(0) {}
        __host__ __device__ LineSegment(const vec2& v0, const vec2& v1) noexcept :
            flags(0), v(v0), dv(v1 - v0) {}

        float Evaluate(const vec2& p, const float& thickness, const float& dPdXY) const;

        __host__ __device__ __forceinline__ BBox2f GetBoundingBox() const
        {
            return BBox2f(vec2(min(v.x, v.x + dv.x), min(v.x, v.y + dv.y)),
                          vec2(max(v.x, v.x + dv.x), max(v.x, v.y + dv.y)));
        }

        uchar flags;
        vec2 v, dv;
    };
    
    struct GI2DOverlayParams
    {
        __host__ __device__ GI2DOverlayParams();

        mat3 viewMatrix;        
        BBox2f sceneBounds;
        float viewScale;

        int selectedSegmentIdx;
        vec2 mousePosView;

        struct
        {
            bool show;
            float majorLineSpacing;
            float minorLineSpacing;
            float lineAlpha;
        } 
        grid;
    };

    namespace Device
    {
        class GI2DOverlay : public Device::Asset
        {
        public:
            struct Objects
            {
                Device::Array<LineSegment>* lineSegments = nullptr;
            };

        public:
            __host__ __device__ GI2DOverlay(const GI2DOverlayParams& params);
            
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
            GI2DOverlay(const std::string& id);
            virtual ~GI2DOverlay();

            __host__ void Render(AssetHandle<Host::ImageRGBA>& hostOutputImage);
            __host__ void OnDestroyAsset();
            __host__ void SetParams(const GI2DOverlayParams& newParams);
            __host__ void UpdateLineSegments(const std::list<LineSegment>& segments);

        private:
            GI2DOverlayParams               m_params;
            Device::GI2DOverlay::Objects    m_objects;
            Device::GI2DOverlay*            cu_deviceData = nullptr;

            //AssetHandle<Host::BIH2D>                    m_hostBIH2D;
            AssetHandle<Host::Array<LineSegment>>       m_hostLineSegments;
        };
    }
}