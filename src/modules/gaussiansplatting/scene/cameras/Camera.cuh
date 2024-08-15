#pragma once

#include "core/2d/Ctx.cuh"
#include "core/3d/Ctx.cuh"
#include "core/3d/Transform.cuh"
#include "core/3d/Basis.cuh"

#include "../SceneObject.cuh"

namespace Enso
{
    struct CameraParams
    {
        __host__ __device__ CameraParams() {}

        __device__ void Validate() const
        {
            CudaAssert(fabsf(trace(fwd)) > 1e-10f);
        }

        vec3                            cameraPos;
        vec3                            cameraLookAt;
        float                           cameraFov;
        mat3                            fwd;     
        mat3                            inv;
    };
     
    namespace Host { class Tracable; }

    namespace Device
    {
        // This class provides an interface for querying the tracable via geometric operations
        class Camera : public Device::SceneObject
        {
        public:
            __device__ virtual bool     CreateRay(const vec2& uvView, const vec2& xi, Ray& ray) const = 0;
            __device__ virtual vec2     ProjectPoint(vec3 p) const = 0;

            __device__ __forceinline__ const CameraParams& GetCameraParams() const { return m_params; }

            __device__ void             Synchronise(const CameraParams& params) { m_params = params; }

        protected:
            __device__                  Camera() {}

            CameraParams                m_params;
        };
    }

    namespace Host
    {        
        class Camera : public Host::SceneObject
        {
        public:
            __host__ virtual void       Synchronise(const uint syncFlags) override final;
            __host__ void               SetPosition(const vec3& cameraPos);
            __host__ Device::Camera*    GetDeviceInstance() { return cu_deviceInstance; }

        protected:
            __host__                    Camera(const Asset::InitCtx& initCtx) : SceneObject(initCtx) {}
            __host__ void               Prepare(const vec3& cameraPos, const vec3& lookAt, const float fov);
            __host__ void               SetDeviceInstance(Device::Camera* deviceInstance) { cu_deviceInstance = deviceInstance; }            
            __host__ virtual void       OnSynchroniseCamera(const uint syncFlags) = 0;

        protected:
            Device::Camera*             cu_deviceInstance;
            CameraParams                m_params;
        };
    }
}