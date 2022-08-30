#pragma once

#include "UILayer.cuh"

#include "../BIH2DAsset.cuh"
#include "../tracables/Tracable.cuh"
#include "../Transform2D.cuh"

using namespace Cuda;

namespace GI2D
{     
    struct PathTracerParams
    {
        __host__ __device__ PathTracerParams();

        UIViewCtx           viewCtx;

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
                Cuda::Device::Vector<GI2D::TracableInterface*>* tracables = nullptr;
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
        class PathTracer : public UILayer
        {
        public:
            PathTracer(const std::string& id, AssetHandle<Host::BIH2DAsset>& bih, AssetHandle<Cuda::Host::AssetVector<Host::Tracable>>& tracables, 
                       const uint width, const uint height, const uint downsample, cudaStream_t renderStream);
            virtual ~PathTracer();
           
            __host__ virtual void Render() override final;
            __host__ virtual void Composite(AssetHandle<Cuda::Host::ImageRGBA>& hostOutputImage) const override final;
            __host__ void OnDestroyAsset();

        protected:
            __host__ virtual void RebuildImpl() override final;

        private:
            PathTracerParams                        m_params;
            GI2D::Device::PathTracer::Objects       m_objects;
            GI2D::Device::PathTracer*               cu_deviceData = nullptr;

            AssetHandle<Cuda::Host::ImageRGBW>       m_hostAccumBuffer;      
        };
    }
}