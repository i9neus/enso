#pragma once

#include "core/3d/Ctx.cuh"
#include "../SceneObject.cuh"

namespace Enso
{
    namespace Device
    {
        // This class provides an interface for querying the tracable via geometric operations
        class Texture2D : public Device::SceneObject
        {
        public:
            __device__ virtual vec4             Evaluate(const vec2& uv) const = 0;
            __device__ __forceinline__ vec4     Evaluate(const HitCtx& hit) const { return Evaluate(hit.uv); }
        };
    }

    namespace Host
    {
        class Texture2D : public Host::SceneObject
        {
        public:
            __host__ Device::Texture2D* GetDeviceInstance() { return cu_deviceInstance; }

        protected:
            __host__                    Texture2D(const Asset::InitCtx& initCtx) : SceneObject(initCtx) {}
            __host__ void               SetDeviceInstance(Device::Texture2D* deviceInstance) { cu_deviceInstance = deviceInstance; }

        protected:
            Device::Texture2D* cu_deviceInstance = nullptr;
        };
    }
}