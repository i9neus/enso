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
    struct PathTracerParams
    {
        __host__ __device__ PathTracerParams();

        GI2D::ViewTransform view;        
        BBox2f sceneBounds;

        struct
        {
            uint width;
            uint height;
            int downsample;
        }
        accum;

        bool isDirty;
        int frameIdx;
    };

    namespace Device
    {
        class PathTracer : public Cuda::Device::Asset
        {
        public:
            struct Objects
            {
                Cuda::Device::Vector<LineSegment>* lineSegments = nullptr;
                GI2D::Device::BIH2DAsset* bih = nullptr;
                Cuda::Device::ImageRGBW* accumBuffer = nullptr;
            };

        public:
            __host__ __device__ PathTracer(const PathTracerParams& params, const Objects& objects);

            __device__ void Synchronise(const PathTracerParams& params);
            __device__ void Synchronise(const Objects& params);
            __device__ void Render();
            __device__ void Composite(Cuda::Device::ImageRGBA* outputImage);

        private:
            PathTracerParams   m_params;
            Objects                m_objects;
        };
    }

    namespace Host
    {
        class PathTracer : public Cuda::Host::Asset
        {
        public:
            PathTracer(const std::string& id, AssetHandle<Host::BIH2DAsset>& bih, AssetHandle<Cuda::Host::Vector<GI2D::LineSegment>>& lineSegments, 
                       const uint width, const uint height, const uint downsample, cudaStream_t renderStream);
            virtual ~PathTracer();
           
            __host__ void Render();
            __host__ void Composite(AssetHandle<Cuda::Host::ImageRGBA>& hostOutputImage);
            __host__ void OnDestroyAsset();
            __host__ void SetParams(const GI2D::PathTracerParams& newParams);
            __host__ void SetDirty() { m_isDirty = true; }

        private:
            PathTracerParams               m_params;
            GI2D::Device::PathTracer::Objects    m_objects;
            GI2D::Device::PathTracer*            cu_deviceData = nullptr;

            AssetHandle<GI2D::Host::BIH2DAsset>               m_hostBIH2D;
            AssetHandle<Cuda::Host::Vector<LineSegment>>      m_hostLineSegments;

            AssetHandle<Cuda::Host::ImageRGBW>       m_hostAccumBuffer;      

            bool                            m_isDirty;
        };
    }
}