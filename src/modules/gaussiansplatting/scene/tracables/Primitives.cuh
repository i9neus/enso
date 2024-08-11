#pragma once

#include "Tracable.cuh"

namespace Enso
{    
    namespace Host { class Tracable; }

    struct PlaneParams
    {
        __device__ void Validate() const {}
        bool isBounded;
    };

    struct UnitSphereParams
    {
        __device__ void Validate() const {}
    };

    namespace Device
    {        
        template<typename ParamsType>
        class Primitive : public Device::Tracable
        {
        public:
            __device__                      Primitive() {}
            __device__ virtual              ~Primitive() {}
            __device__ virtual bool         IntersectRay(Ray& ray, HitCtx& hit) const override final; 
            __device__ void                 Synchronise(const ParamsType& params) { m_params = params; }

        private:
            ParamsType                      m_params;
        };
    }

    namespace Host
    {        
        template<typename ParamsType>
        class Primitive : public Host::Tracable
        {
            using DeviceType = Device::Primitive<ParamsType>;
        public:
            __host__                Primitive(const InitCtx& initCtx);
            __host__                Primitive(const InitCtx& initCtx, const BidirectionalTransform& transform, const int materialIdx, const ParamsType& params);
            __host__ virtual        ~Primitive();

            __host__ virtual void   OnSynchroniseTracable(const uint syncFlags) override final
            {
                SynchroniseObjects<DeviceType>(cu_deviceInstance, m_params);
            }

        private:
            DeviceType*           cu_deviceInstance;
            ParamsType            m_params;
        };       

        template class Primitive<PlaneParams>;
        template class Primitive<UnitSphereParams>;
    }
}