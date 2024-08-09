#pragma once

#include "Camera.cuh"

namespace Enso
{
    struct PinholeCameraParams
    {
        __host__ __device__ PinholeCameraParams() :
            fov(45.) {}

        __device__ void Validate() const {}

        float fov;
    };
     
    namespace Device
    {
        // This class provides an interface for querying the tracable via geometric operations
        class PinholeCamera : public Device::Camera
        {
        public:
            __device__ PinholeCamera() {}

            __device__ virtual bool CreateRay(const vec2& uvView, const vec2& xi, Ray& ray) const override final
            {
                // uvView should be in the range [-0.5, 0.5]

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

            __host__ PinholeCamera(const Asset::InitCtx& initCtx, const vec3& cameraPos, const vec3& cameraBasis, const float fov) :
                PinholeCamera(initCtx)
            {
                Camera::Prepare(cameraPos, cameraBasis);
                m_params.fov = fov;                                
                
                Synchronise(kSyncParams);
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

            PinholeCameraParams                m_params;
        };
    }
}