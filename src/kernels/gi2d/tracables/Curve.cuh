#pragma once

#include "Tracable.cuh"

#define FWD_DECL_VECTOR
#define FWD_DECL_BIH2D
#include "../FwdDecl.cuh"

using namespace Cuda;

namespace GI2D
{
    class LineSegment;

    struct CurveObjects
    {
        __host__ __device__ CurveObjects() {}

        BIH2D<BIH2DFullNode>* m_bih = nullptr;
        VectorInterface<LineSegment>* m_lineSegments = nullptr;
    };

    namespace Device
    {        
        class Curve : public TracableInterface,                           
                      public CurveObjects
        {
        public:
            __device__ Curve() {}

            __host__ __device__ virtual bool    IntersectRay(Ray2D& ray, HitCtx2D& hit) const override final;

        protected:
            __device__ virtual bool             EvaluatePrimitives(const vec2& pWorld, const UIViewCtx& viewCtx, vec4& L) const override final;
        };
    }

    namespace Host
    {                  
        class BIH2DAsset;
        
        class Curve : public Host::Tracable,
                      public CurveObjects,
                      public Cuda::AssetTags<Host::Curve, Device::Curve>
        {
        public:
            __host__ Curve(const std::string& id);
            __host__ virtual ~Curve();

            __host__ virtual void       OnDestroyAsset() override final;
            __host__ void               Synchronise(const int syncType);

            __host__ virtual uint       OnCreate(const std::string& stateID, const UIViewCtx& viewCtx) override final;
            //__host__ virtual uint       OnSelectElement(const std::string& stateID, const vec2& mousePos, const UIViewCtx& viewCtx, UISelectionCtx& selectCtx) override final;
            __host__ virtual bool       IsConstructed() const override final;
            __host__ virtual bool       Rebuild(const uint parentFlags, const UIViewCtx& viewCtx) override final;
            __host__ virtual bool       Finalise() override final;

            __host__ Device::Curve* GetDeviceInstance() const
            { 
                return cu_deviceInstance;
            }

            __host__ static AssetHandle<GI2D::Host::SceneObject> Instantiate(const std::string& id);
            __host__ static const std::string GetAssetTypeString() { return "curve"; }

            __host__ static void Test();

        private:


        private:
            Device::Curve*                                  cu_deviceInstance = nullptr;
            CurveObjects                                    m_deviceObjects;

            AssetHandle<Host::BIH2DAsset>                   m_hostBIH;
            AssetHandle<Cuda::Host::Vector<LineSegment>>    m_hostLineSegments;

            int                                             m_numSelected;

        };
    }
}