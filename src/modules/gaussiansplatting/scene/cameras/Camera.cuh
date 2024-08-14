#pragma once

#include "core/2d/Ctx.cuh"
#include "core/3d/Ctx.cuh"
#include "core/GenericObject.cuh"
#include "core/3d/Transform.cuh"
#include "core/3d/Basis.cuh"

namespace Enso
{
    struct CameraParams
    {
        __host__ __device__ CameraParams() {}

        __device__ void Validate() const
        {
            CudaAssert(fabsf(trace(fwd)) > 1e-10f);
        }

        vec3 cameraPos;
        vec3 cameraLookAt;
        float cameraFov;
        mat3 fwd;     
        mat3 inv;
    };
     
    namespace Host { class Tracable; }

    namespace Device
    {
        // This class provides an interface for querying the tracable via geometric operations
        class Camera : public Device::GenericObject
        {
        public:
            __device__ virtual bool                 CreateRay(const vec2& uvView, const vec2& xi, Ray& ray) const = 0;
            __device__ virtual vec2                 ProjectPoint(vec3 p) const = 0;

            __device__ __forceinline__ const CameraParams& GetCameraParams() const { return m_params; }

            __device__ void                         Synchronise(const CameraParams& params) { m_params = params; }

        protected:
            __device__ Camera() {}

            CameraParams m_params;
        };
    }

    namespace Host
    {        
        class Camera : public Host::GenericObject
        {
        public:
            __host__ virtual void Synchronise(const uint syncFlags) override final
            {
                if (syncFlags & kSyncParams)
                {
                    SynchroniseObjects<Device::Camera>(cu_deviceInstance, m_params);
                }
                OnSynchroniseCamera(syncFlags);
            }

            __host__ void Prepare(const vec3& cameraPos, const vec3& lookAt, const float fov)
            {
                m_params.cameraPos = cameraPos;
                m_params.cameraLookAt = lookAt;
                m_params.cameraFov = fov;
                m_params.inv = CreateBasis(normalize(m_params.cameraPos - m_params.cameraLookAt), vec3(0.f, 1.f, 0.f));
                m_params.fwd = transpose(m_params.inv);
            }

            __host__ Device::Camera*    GetDeviceInstance() { return cu_deviceInstance; }

        protected:
            __host__                    Camera(const Asset::InitCtx& initCtx) : GenericObject(initCtx) {}
            __host__ void               SetDeviceInstance(Device::Camera* deviceInstance) { cu_deviceInstance = deviceInstance; }            
            __host__ virtual void       OnSynchroniseCamera(const uint syncFlags) = 0;

        protected:
            Device::Camera*             cu_deviceInstance;
            CameraParams                m_params;
        };
    }
}