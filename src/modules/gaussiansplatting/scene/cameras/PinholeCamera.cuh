#pragma once

#include "Camera.cuh"

namespace Enso
{
    struct PinholeCameraParams
    {
        __host__ __device__ PinholeCameraParams() :
            fov(45.) {}

        float fov;
    };
     
    namespace Device
    {
        // This class provides an interface for querying the tracable via geometric operations
        class PinholeCamera : public Device::Camera
        {
            friend Host::PinholeCamera;

        public:
            __device__ virtual bool CreateRay(const ivec2& xyViewport, const vec4& xi, Ray& ray)
            {
                const vec2 uvView = ScreenToNormalisedScreen(vec2(xyViewport) + xi.xy, Camera::m_params.viewportDims);
                
                ray.od.o = Camera::m_params.cameraPos;
                ray.od.d = Camera::m_params.cameraBasis * normalize(vec3(uvView, -tanf(toRad(PinholeCamera::m_params.fov))));
                ray.tNear = kFltMax;
                ray.weight = vec3(1.0, 1.0, 1.0);
                ray.flags = kRayCausticPath;
                ray.depth = 0;

                return true;
            }

            __device__ void  Synchronise(const PinholeCameraParams& params) { m_params = params; }

        protected:
            __device__ PinholeCamera() {}

            PinholeCameraParams m_params;
        };
    }

    namespace Host
    {        
        class PinholeCamera : public Host::Camera
        {
        public:
            __host__ PinholeCamera(const Asset::InitCtx& initCtx) :
                Camera(initCtx),
                cu_deviceInstance(AssetAllocator::InstantiateOnDevice<Device::PinholeCamera>(*this))
            {
                Camera::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::Camera>(cu_deviceInstance));
            }

            __host__ virtual ~PinholeCamera() noexcept
            {
                AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);
            }

        protected:            
            __host__ virtual void  OnSynchroniseCamera(const uint syncFlags) override final
            {
                if (syncFlags & kSyncParams)
                {
                    SynchroniseObjects<Device::PinholeCamera>(cu_deviceInstance, m_params);
                }
            }

        protected:
            Device::PinholeCamera*             cu_deviceInstance;

            CameraParams                m_params;
        };
    }
}