#pragma once

#include "Tracable.cuh"
#include "../FwdDecl.cuh"

namespace Enso
{
    class LineSegment;

    struct CurveObjects
    {
        __host__ __device__ CurveObjects() {}

        BIH2D<BIH2DFullNode>* m_bih = nullptr;
        Generic::Vector<LineSegment>* m_lineSegments = nullptr;

        __host__ __device__ CurveObjects& operator=(const CurveObjects& other)
        {
            m_bih = other.m_bih;
            m_lineSegments = other.m_lineSegments;
            printf("0x%x\n", m_bih);
            return *this;
        }
    };

    namespace Device
    {        
        class Curve : public Device::Tracable,                           
                      public CurveObjects
        {
        public:
            __host__ __device__ Curve() {}

            __host__ __device__ virtual bool    IntersectRay(const Ray2D& ray, HitCtx2D& hit) const override final;

            __device__ virtual vec4             EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx) const override final;
        };
    }

    namespace Host
    {                  
        class BIH2DAsset;
        
        class Curve : public Host::Tracable                     
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

            __host__ virtual Device::Curve* GetDeviceInstance() const override final
            { 
                return cu_deviceInstance;
            }

            __host__ static AssetHandle<Host::GenericObject> Instantiate(const std::string& id, const Json::Node&);
            __host__ static const std::string GetAssetClassStatic() { return "curve"; }
            __host__ virtual std::string GetAssetClass() const override final { return GetAssetClassStatic(); }

            __host__ virtual bool       Serialise(Json::Node& rootNode, const int flags) const override final;
            __host__ virtual uint       Deserialise(const Json::Node& rootNode, const int flags) override final;

        protected:
            __host__ virtual BBox2f     RecomputeObjectSpaceBoundingBox() override final;

        private:
            Device::Curve*                                  cu_deviceInstance = nullptr;
            Device::Curve                                   m_hostInstance;
            CurveObjects                                    m_deviceObjects;

            AssetHandle<Host::BIH2DAsset>                   m_hostBIH;
            AssetHandle<Host::Vector<LineSegment>>          m_hostLineSegments;

            int                                             m_numSelected;
        };
    }

    // Explicitly declare instances of this class for its inherited types
    //template class Host::Tracable<Device::Curve>;
    //template class Host::GenericObject<Device::Curve>;
}