#pragma once

#include "Light.cuh"

namespace Enso
{
    namespace Device
    {
        class QuadLight : public Device::Light
        {
        public:
            __device__      QuadLight() {}
            __device__      ~QuadLight() noexcept {}

            __device__ virtual float        Sample(const Ray& incident, Ray& extant, const HitCtx& hit, const vec2& xi) override final;
            __device__ virtual float        Evaluate(Ray& extant, const HitCtx& hit) override final;
            __device__ virtual bool         IntersectRay(Ray& ray, HitCtx& hit) const override final;
        };
    }

    namespace Host
    {
        class QuadLight : public Host::Light
        {
        public:
            __host__ QuadLight(const Asset::InitCtx& initCtx);
            __host__ virtual ~QuadLight() noexcept;

        protected:
            __host__ virtual void OnSynchroniseLight(const uint syncFlags) override final {}

        private:
            Device::QuadLight* cu_deviceInstance = nullptr;
        };
    }
}