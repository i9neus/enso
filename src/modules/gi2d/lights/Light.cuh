#pragma once

#include "../tracables/Tracable.cuh"

namespace Enso
{
    namespace Host
    {
        class Light;
    }

    namespace Device
    {
        class Light : public Device::Tracable
        {
            friend class Host::Light;

        public:
            __host__ __device__ Light() {}

            __device__ virtual bool                     Sample(const Ray2D& parentRay, const HitCtx2D& hit, float xi, vec2& extant, vec3& L, float& pdf) const = 0;
            __device__ virtual bool                     Evaluate(const Ray2D& parentRay, const HitCtx2D& hit, vec3& L, float& pdfLight) const = 0;
        };
    }

    namespace Host
    {
        class Light : public Host::Tracable
        {
        public:
            __host__ Light(const std::string& id, Device::Light& hostInstance) :
                Tracable(id, hostInstance),
                m_hostInstance(hostInstance)
            {}

            __host__ virtual ~Light() {}

            __host__ virtual Device::Light* GetDeviceInstance() const = 0;

        protected:
            template<typename SubType> __host__ inline void Synchronise(SubType* deviceData, const int syncType) { Tracable::Synchronise(deviceData, syncType); }

        private:
            Device::Light& m_hostInstance;
        };
    }
}