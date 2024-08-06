#pragma once

#include "Tracable.cuh"

namespace Enso
{    
    namespace Host { class Tracable; }

    struct PlanePrimitiveParams
    {
        __device__ void Validate() const {}

        bool isBounded;
    };

    namespace Device
    {
        // This class provides an interface for querying the tracable via geometric operations
        class PlanePrimitive : public Device::Tracable
        {
        public:
            __device__                      PlanePrimitive() {}
            __device__ virtual              ~PlanePrimitive() {}
            __device__ virtual bool         IntersectRay(Ray& ray, HitCtx& hit) const override final; 
            __device__ void                 Synchronise(const PlanePrimitiveParams& params) { m_params = params; }

        private:
            PlanePrimitiveParams        m_params;
        };

        // This class provides an interface for querying the tracable via geometric operations
        class SpherePrimitive : public Device::Tracable
        {
        public:
            __device__                      SpherePrimitive() {}
            __device__ virtual              ~SpherePrimitive() {}
            __device__ virtual bool         IntersectRay(Ray& ray, HitCtx& hit) const override final;
        };
    }

    namespace Host
    {        
        class PlanePrimitive : public Host::Tracable
        {
        public:
            __host__                PlanePrimitive(const InitCtx& initCtx);
            __host__ virtual        ~PlanePrimitive();

            __host__ virtual void   OnSynchroniseTracable(const uint syncFlags) override final
            {
                SynchroniseObjects<Device::PlanePrimitive>(cu_deviceInstance, m_params);
            }

        private:
            Device::PlanePrimitive*         cu_deviceInstance;
            PlanePrimitiveParams            m_params;
        };

        class SpherePrimitive : public Host::Tracable
        {
        public:
            __host__                SpherePrimitive(const InitCtx& initCtx);
            __host__ virtual        ~SpherePrimitive();
            __host__ virtual void   OnSynchroniseTracable(const uint syncFlags) override final {}

        private:
            Device::SpherePrimitive* cu_deviceInstance;
        };
    }
}