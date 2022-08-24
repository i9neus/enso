#pragma once

#include "Common.cuh"

#include "CudaPrimitive2D.cuh"
#include "BIH2DAsset.cuh"
#include "Transform2D.cuh"
#include "UICtx.cuh"

#include "../CudaRenderObject.cuh"

using namespace Cuda;

namespace GI2D
{
    struct TracableParams
    {
        BBox2f                      objectBBox;
        BidirectionalTransform2D    transform;
    };
    
    // This class provides an interface for querying the tracable via geometric operations
    class TracableInterface
    {
    public:
        /*__host__ __device__ virtual bool                    IntersectRay(Ray2D& ray, HitCtx2D& hit, float& tFar) const { return false; }
        __host__ __device__ virtual bool                    InteresectPoint(const vec2& p, const float& thickness) const { return false; }
        __host__ __device__ virtual bool                    IntersectBBox(const BBox2f& bBox) const { return false; }

        __host__ __device__ virtual vec2                    PerpendicularPoint(const vec2& p) const { return vec2(0.0f); }*/
        __device__ virtual vec4                             EvaluateOverlay(const vec2& p, const ViewTransform2D& viewCtx) const { return vec4(0.0f); }

        __host__ __device__ const BBox2f&                   GetBoundingBox() const { return m_tracableParams.objectBBox; };
        __device__ void                                     Synchronise(const TracableParams& params) { m_tracableParams = params; }

    protected:
        __host__ __device__ TracableInterface() {};

        __host__ __device__ __forceinline__ RayBasic2D ToObjectSpace(const Ray2D& world) const
        {
            RayBasic2D obj;
            obj.o = world.o - m_tracableParams.transform.trans;
            obj.d = world.d + obj.o;
            obj.o = m_tracableParams.transform.fwd * obj.o;
            obj.d = (m_tracableParams.transform.fwd * obj.d) - obj.o;
            return obj;
        }

    protected:
        TracableParams m_tracableParams;
    };
    
    namespace Device
    {
        class Tracable : virtual public TracableInterface,
                              public Cuda::Device::RenderObject
        {
        public:
            __device__ Tracable() {}
        };
    }

    namespace Host
    {
        class Tracable : virtual public TracableInterface,
                         public Cuda::Host::RenderObject, 
                         public Cuda::AssetTags<Host::Tracable, TracableInterface>
        {
        public:
            __host__ virtual uint       OnCreate(const std::string& stateID, const UIViewCtx& viewCtx) = 0;
            __host__ virtual uint       OnSelect() = 0;
            //__host__ virtual uint       OnSelectElement(const std::string& stateID, const UIViewCtx& viewCtx, UISelectionCtx& selectCtx) = 0;
            __host__ virtual uint       OnMove(const std::string& stateID) = 0;
            __host__ virtual bool       Finalise() = 0;

            __host__ virtual bool       IsEmpty() const = 0;
            __host__ virtual void       Rebuild() = 0;

            __host__ uint               GetDirtyFlags() const { return m_dirtyFlags; }
            __host__ bool               IsFinalised() const { return m_isFinalised; }

            __host__ virtual TracableInterface* GetDeviceInstance() const = 0;

        protected:
            __host__ Tracable(const std::string& id) : RenderObject(id), m_dirtyFlags(0), m_isFinalised(false) {}

            __host__ void               SetDirtyFlags(const uint flags) { m_dirtyFlags |= flags; }
            __host__ void               ClearDirtyFlags(const uint flags) { m_dirtyFlags &= ~flags; }

            uint                        m_dirtyFlags;
            bool                        m_isFinalised;
        };
    }
}