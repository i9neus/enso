#pragma once

#include "CudaPrimitive2D.cuh"
#include "BIH2DAsset.cuh"
#include "Transform2D.cuh"

using namespace Cuda;

namespace GI2D
{
    struct CurveParams
    {
    };

    namespace Device
    {
        class Tracable
        {
        public:
            __device__ virtual bool Intersect(Ray2D& ray, HitCtx2D& hit, float& tFar) const = 0;

        protected:
            __device__ __forceinline__ RayBasic2D ToObjectSpace(const Ray2D& world) const
            {
                RayBasic2D obj;
                obj.o = world.o - m_transform.trans;
                obj.d = world.d + obj.o;
                obj.o = m_transform.fwd * obj.o;
                obj.d = (m_transform.fwd * obj.d) - obj.o;
                return obj;
            }

        protected:
            BidirectionalTransform2D m_transform;
        };
        
        class Curve : public Tracable, public Cuda::Device::Asset
        {
        public:
            struct Objects
            {
                Device::BIH2DAsset* bih = nullptr;
                Cuda::Device::Vector<LineSegment>* lineSegments = nullptr;
            };

        public:
            __device__ Curve();

            __device__ virtual bool     Intersect(Ray2D& ray, HitCtx2D& hit, float& tFar) const override final;
            __device__ void             Synchronise(const Objects& objects);

        private:
            Objects                     m_objects;
        };
    }

    namespace Host
    {
        class Curve : public Cuda::Host::Asset
        {
        public:
            __host__ Curve(const std::string& id);
            __host__ virtual ~Curve();

            __host__ virtual void       OnDestroyAsset() override final;
            __host__ void               Synchronise();
        private:
           

        private:
            Device::Curve*              cu_deviceInstance;
            Device::Curve::Objects      m_deviceData;

            AssetHandle<Host::BIH2DAsset>                   m_hostBIH;
            AssetHandle<Cuda::Host::Vector<LineSegment>>    m_hostLineSegments;
        };
    }
}