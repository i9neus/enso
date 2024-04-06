#pragma once

#include "Tracable.cuh"
#include "../FwdDecl.cuh"

namespace Enso
{
    class LineSegment;

    struct LineStripObjects
    {
        __host__ __device__ LineStripObjects() {}

        BIH2D<BIH2DFullNode>*           bih = nullptr;
        Generic::Vector<LineSegment>*   lineSegments = nullptr;
    };

    namespace Host { class LineStrip; }
    
    namespace Device
    {
        class LineStrip : public Device::Tracable
        {
            friend class Host::LineStrip;

        public:
            __host__ __device__ LineStrip() {}

            __host__ __device__ virtual bool    IntersectRay(const Ray2D& ray, HitCtx2D& hit) const override final;
            __host__ __device__ virtual vec4    EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx) const override final;
            __host__ __device__ virtual uint    OnMouseClick(const UIViewCtx& viewCtx) const override final;
            __host__ __device__ void            Print() const;

            __device__ void                     Synchronise(const LineStripObjects& objects) { m_objects = objects; }

        private:
            LineStripObjects m_objects;
        };
    }

    namespace Host
    {
        class BIH2DAsset;

        class LineStrip : public Host::Tracable
        {
        public:
            __host__ LineStrip(const std::string& id);
            __host__ virtual ~LineStrip();

            __host__ virtual void       OnDestroyAsset() override final;
            __host__ void               Synchronise(const int syncType);

            __host__ virtual uint       OnCreate(const std::string& stateID, const UIViewCtx& viewCtx) override final;
            __host__ virtual uint       OnMouseClick(const UIViewCtx& viewCtx) const override final;

            //__host__ virtual uint       OnSelectElement(const std::string& stateID, const vec2& mousePos, const UIViewCtx& viewCtx, UISelectionCtx& selectCtx) override final;
            __host__ virtual bool       IsConstructed() const override final;
            __host__ virtual bool       Rebuild(const uint parentFlags, const UIViewCtx& viewCtx) override final;

            __host__ virtual Device::LineStrip* GetDeviceInstance() const override final
            {
                return cu_deviceInstance;
            }

            __host__ static AssetHandle<Host::GenericObject> Instantiate(const std::string& id, const Json::Node&, const AssetHandle<const Host::SceneDescription>&);
            __host__ static const std::string GetAssetClassStatic() { return "curve"; }
            __host__ virtual std::string GetAssetClass() const override final { return GetAssetClassStatic(); }

            __host__ virtual bool       Serialise(Json::Node& rootNode, const int flags) const override final;
            __host__ virtual uint       Deserialise(const Json::Node& rootNode, const int flags) override final;

        protected:
            __host__ virtual BBox2f     RecomputeObjectSpaceBoundingBox() override final;

        private:
            Device::LineStrip*                              cu_deviceInstance = nullptr;
            Device::LineStrip                               m_hostInstance;
            LineStripObjects                                m_deviceObjects;

            AssetHandle<Host::BIH2DAsset>                   m_hostBIH;
            AssetHandle<Host::Vector<LineSegment>>          m_hostLineSegments;

            int                                             m_numSelected;
        };
    }

    // Explicitly declare instances of this class for its inherited types
    //template class Host::Tracable<Device::LineStrip>;
    //template class Host::GenericObject<Device::LineStrip>;
}