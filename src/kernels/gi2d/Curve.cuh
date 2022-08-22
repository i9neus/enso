#pragma once

#include "Tracable.cuh"

using namespace Cuda;

namespace GI2D
{
    struct CurveParams
    {
    };

    class CurveInterface : virtual public TracableInterface
    {
    public:
        __host__ __device__ CurveInterface() : m_bih(nullptr), m_lineSegments(nullptr) {}

        /*__host__ __device__ virtual bool     IntersectRay(Ray2D& ray, HitCtx2D& hit, float& tFar) const override final;
        __host__ __device__ virtual bool     InteresectPoint(const vec2& p, const float& thickness) const override final;
        __host__ __device__ virtual bool     IntersectBBox(const BBox2f& bBox) const override final;

        __host__ __device__ virtual vec2     PerpendicularPoint(const vec2& p) const override final;
        __host__ __device__ virtual float    Evaluate(const vec2& p, const float& thickness, const float& dPdXY) const override final;*/

    protected:
        BIH2D<BIH2DFullNode>*               m_bih;
        VectorInterface<LineSegment>*       m_lineSegments;
    };

    namespace Device
    {
        class Curve : public GI2D::CurveInterface // , public Cuda::AssetTags<Host::Curve, Device::Curve>
        {
        public:
            struct Objects
            {
                Device::BIH2DAsset* bih = nullptr;
                Cuda::Device::Vector<LineSegment>* lineSegments = nullptr;
            };

        public:
            __device__ Curve() {}

            __device__ void             Synchronise(const Objects& objects);
        };
    }

    namespace Host
    {                  
        class Curve : public CurveInterface,
                      public Host::Tracable,
                      public Cuda::AssetTags<Host::Curve, Device::Curve>
        {
        public:
            __host__ Curve(const std::string& id);
            __host__ virtual ~Curve();

            __host__ virtual void       OnDestroyAsset() override final;
            __host__ void               Synchronise();

            __host__ virtual uint       OnCreate(const std::string& stateID, const UIViewCtx& viewCtx) override final;
            __host__ virtual uint       OnSelect() override final { return 0; }
            //__host__ virtual uint       OnSelectElement(const std::string& stateID, const vec2& mousePos, const UIViewCtx& viewCtx, UISelectionCtx& selectCtx) override final;
            __host__ virtual uint       OnMove(const std::string& stateID) override final { return 0; }
            __host__ virtual bool       IsEmpty() const override final;
            __host__ virtual void       Rebuild() override final;
        private:


        private:
            Device::Curve*                                  cu_deviceInstance;
            Device::Curve::Objects                          m_deviceData;

            AssetHandle<Host::BIH2DAsset>                   m_hostBIH;
            AssetHandle<Cuda::Host::Vector<LineSegment>>    m_hostLineSegments;

            int                                             m_numSelected;

        };
    }
}