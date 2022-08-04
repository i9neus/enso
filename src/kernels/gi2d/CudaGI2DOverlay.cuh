#pragma once

#include "../CudaAsset.cuh"
#include "../CudaManagedObject.cuh"
#include "../CudaImage.cuh"
#include "../CudaManagedArray.cuh"

namespace Cuda
{
    __host__ __device__ __forceinline__ vec2 TransformSceenToView2D(const mat3& m, const vec2& p)
    {
        return (m * vec3(p, 1.0f)).xy;
    }
    
    struct GI2DOverlayParams
    {
        __host__ __device__ GI2DOverlayParams();

        mat3 viewMatrix;
        
        float viewScale;
        float majorLineSpacing;
        float minorLineSpacing;
        float lineAlpha;
    };

    namespace Device
    {
        class GI2DOverlay : public Device::Asset
        {
        public:
            __host__ __device__ GI2DOverlay(const GI2DOverlayParams& params);
            
            __device__ void Synchronise(const GI2DOverlayParams& params);
            __device__ void Render(Device::ImageRGBA* outputImage);

        private:
            GI2DOverlayParams m_params;
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

        private:
            GI2DOverlayParams       m_params;
            Device::GI2DOverlay*    cu_deviceData = nullptr;
        };
    }
}