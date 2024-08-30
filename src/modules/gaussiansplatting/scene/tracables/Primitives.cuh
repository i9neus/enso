#pragma once

#include "Tracable.cuh"

namespace Enso
{    
    namespace Host { class Tracable; }

    struct PlaneParams
    {
        __host__ __device__ PlaneParams() : isBounded(true) {}
        __host__ __device__  PlaneParams(const bool b) : isBounded(b) {}
        __device__ void Validate() const {}

        bool isBounded;
    };

    struct UnitSphereParams
    {
        __device__ void Validate() const {}
    };

    struct CylinderParams
    {
        __host__ __device__ CylinderParams() : height(1) {}
        __host__ __device__ CylinderParams(const float h) : height(h) {}
        __device__ void Validate() const 
        {
            CudaAssert(height > 0.);
        } 

        float height;
    };

    struct BoxParams
    {
        __host__ __device__ BoxParams() : dims(1) {}
        __host__ __device__ BoxParams(const vec3& d) : dims(d) {}
        __device__ void Validate() const
        {
            CudaAssert(Volume(dims) > 0.f);
        }

        vec3 dims;
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
            __host__                Primitive(const InitCtx& initCtx, const BidirectionalTransform& transform, const int materialIdx, const ParamsType& params);
            __host__ virtual        ~Primitive();

            __host__ virtual std::vector<GaussianPoint> GenerateGaussianPointCloud(const int numPoints, const float areaGain, MersenneTwister& rng) override final;
            __host__ virtual float  CalculateSurfaceArea() const override final;

            __host__ virtual void   OnSynchroniseTracable(const uint syncFlags) override final
            {
                if (syncFlags & kSyncParams)
                {
                    SynchroniseObjects<DeviceType>(cu_deviceInstance, m_params);
                }
                OnSynchronisePrimitive(syncFlags);
            }

        protected:
            __host__ void           SetDeviceInstance(Device::Primitive<ParamsType>* deviceInstance) { cu_deviceInstance = deviceInstance; }
            __host__ virtual void   OnSynchronisePrimitive(const uint syncFlags) {};


        private:
            __host__ GaussianPoint GenerateRandomGaussianPoint(const vec3& p, float gaussSigma, MersenneTwister& rng) const;

        private:
            DeviceType*           cu_deviceInstance;
            ParamsType            m_params;
        };       

        template class Primitive<PlaneParams>;
        template class Primitive<UnitSphereParams>;
        template class Primitive<CylinderParams>;
        template class Primitive<BoxParams>;

        using PlanePrimitive = Primitive<PlaneParams>; 
        using SpherePrimitive = Primitive<UnitSphereParams>;
        using CylinderPrimitive = Primitive<CylinderParams>;
        using BoxPrimitive = Primitive<BoxParams>;
    }
}