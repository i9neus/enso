#pragma once

#include "../SceneObject.cuh"
#include "core/3d/Ctx.cuh"

namespace Enso
{
    struct MaterialParams
    {
        __device__ void Validate() const {}

        int textureIdx = -1;
    };

    namespace Device
    {
        // This class provides an interface for querying the tracable via geometric operations
        class Material : public Device::SceneObject
        {
        public:
            __device__ virtual float    Sample(const vec2& xi, const vec3& i, const vec3& n, vec3& o, float& weight) const = 0;
            __device__ virtual float    Evaluate(const vec3& i, const vec3& o, const vec3& n) const = 0;
            __device__ virtual bool     IsPerfectSpecular() const = 0;

            __device__ void             Synchronise(const MaterialParams& params) { m_params = params; }

        protected:
            MaterialParams              m_params;
        };
    }

    namespace Host
    {
        class Material : public Host::SceneObject
        {
        public:
            __host__ Device::Material* GetDeviceInstance() { return cu_deviceInstance; }

        protected:
            __host__ Material(const Asset::InitCtx& initCtx, const int textureIdx) :
                SceneObject(initCtx)
            {
                m_params.textureIdx = textureIdx;
            }
            
            __host__ void               SetDeviceInstance(Device::Material* deviceInstance) { cu_deviceInstance = deviceInstance; }

            __host__ void Synchronise(const int syncFlags)
            {
                if (syncFlags & kSyncParams)
                {
                    SynchroniseObjects<Device::Material>(cu_deviceInstance, m_params);
                }
            }

        protected:
            Device::Material*           cu_deviceInstance = nullptr;

            MaterialParams              m_params;
        };
    }
}