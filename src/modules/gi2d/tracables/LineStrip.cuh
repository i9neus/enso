#pragma once

#include "Tracable.cuh"
#include "../FwdDecl.cuh"

namespace Enso
{
    class LineSegment;

    struct LineStripObjects
    {
        __host__ __device__ LineStripObjects() {}

        __device__ void Validate() const
        {
            CudaAssert(bih);
            CudaAssert(lineSegments);
        }

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
            __host__ __device__ virtual vec4    EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx, const bool isMouseTest) const override final;
            __host__ __device__ uint            OnMouseClick(const UIViewCtx& viewCtx) const;
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
            __host__ LineStrip(const Asset::InitCtx& initCtx);
            __host__ virtual ~LineStrip() noexcept;

            __host__ virtual bool       OnRebuildSceneObject() override final;
            __host__ virtual void       OnSynchroniseTracable(const uint syncType) override final;
            __host__ virtual uint       OnMouseClick(const UIViewCtx& viewCtx) const override final;

            //__host__ virtual uint       OnSelectElement(const std::string& stateID, const vec2& mousePos, const UIViewCtx& viewCtx, UISelectionCtx& selectCtx) override final;
            __host__ virtual bool       IsConstructed() const override final;

            __host__ virtual Device::LineStrip* GetDeviceInstance() const override final
            {
                return cu_deviceInstance;
            }

            __host__ static AssetHandle<Host::GenericObject> Instantiate(const std::string& id, const Host::Asset& parentAsset, const AssetHandle<const Host::SceneContainer>& scene);
            __host__ static const std::string GetAssetClassStatic() { return "curve"; }
            __host__ virtual std::string GetAssetClass() const override final { return GetAssetClassStatic(); }

            __host__ virtual bool       Serialise(Json::Node& rootNode, const int flags) const override final;
            __host__ virtual bool       Deserialise(const Json::Node& rootNode, const int flags) override final;

            __host__ virtual BBox2f     ComputeObjectSpaceBoundingBox() override final;

        protected:
            __host__ virtual bool       OnCreateSceneObject(const std::string& stateID, const UIViewCtx& viewCtx, const vec2& mousePosObject) override final;

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