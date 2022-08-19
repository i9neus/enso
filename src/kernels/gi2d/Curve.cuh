#pragma once

#include "Tracable.cuh"

using namespace Cuda;

namespace GI2D
{
    struct CurveParams
    {
    };

    namespace Device
    {
        class Curve : public Tracable
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
        class Curve : public Tracable
        {
        public:
            __host__ Curve(const std::string& id);
            __host__ virtual ~Curve();

            __host__ virtual void       OnDestroyAsset() override final;
            __host__ void               Synchronise();

            __host__ virtual uint       OnCreate(const std::string& stateID, const vec2& mousePos) override final;
            __host__ virtual uint       OnSelect() = 0;
            __host__ virtual uint       OnDeselect() = 0;
            __host__ virtual uint       OnSelectElement(const std::string& stateID, const vec2& mousePos, const UIViewCtx& viewCtx, UISelectionCtx& selectCtx) override final;
        private:
            

        private:
            Device::Curve*              cu_deviceInstance;
            Device::Curve::Objects      m_deviceData;

            AssetHandle<Host::BIH2DAsset>                   m_hostBIH;
            AssetHandle<Cuda::Host::Vector<LineSegment>>    m_hostLineSegments;

            int                         m_numSelected;

        };
    }
}