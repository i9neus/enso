#pragma once

#include "core/2d/Ctx.cuh"
#include "core/2d/DrawableObject.cuh"
#include "core/3d/Transform.cuh"

#include "core/DirtinessFlags.cuh"
#include "core/Image.cuh"
#include "core/GenericObject.cuh"
#include "core/HighResolutionTimer.h"

#include "../../FwdDecl.cuh"

#include "../tracables/Tracable.cuh"

namespace Enso
{
    struct LightParams
    {
        float radiance;
    };
    
    namespace Device
    {
        class Light : public Device::Tracable
        {
        public:
            __device__      Light() {}
            __device__      ~Light() noexcept {}
            __device__      void Synchronise(const LightParams& params) { m_params = params; }
            __device__      void Verify() {}

            __device__ virtual float Sample(const Ray& incident, Ray& extant, const HitCtx& hit, const vec2& xi) = 0;
            __device__ virtual float Evaluate(Ray& extant, const HitCtx& hit) = 0;


        protected:
            LightParams     m_params;
        };
    }

    namespace Host
    {
        class Light : public Host::Tracable
        {
        public:
            __host__ virtual ~Light() noexcept {}

        protected:
            __host__ Light(const Asset::InitCtx& initCtx) :
                Tracable(initCtx)
            {}

            __host__ void SetDeviceInstance(Device::Light* deviceInstance)
            {
                Tracable::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::Tracable>(deviceInstance));
                cu_deviceInstance = deviceInstance;
            }

        private:
            Device::Light* cu_deviceInstance = nullptr;
        };
    }
}