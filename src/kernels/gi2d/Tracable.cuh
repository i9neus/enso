#pragma once

#include "CudaPrimitive2D.cuh"
#include "BIH2DAsset.cuh"
#include "Transform2D.cuh"
#include "UICtx.cuh"

#include "../CudaRenderObject.cuh"

using namespace Cuda;

namespace GI2D
{
    enum GI2DDirtyFlags : uint
    {
        kGI2DClean = 0,
        
        // View params i.e. camera position and orientation
        kGI2DDirtyView = 1,

        // UI changes like selection and lassoing
        kGI2DDirtyUI = 2,

        // Primitive attributes that don't affect the hierarchy like material properteis
        kGI2DDirtyPrimitiveAttributes = 4,

        // Changes to geometry that require a complete rebuild of the hierarchy
        kGI2DDirtyGeometry = 8,

        // Changes to the number of scene objects
        kGI2DDirtySceneObjectCount = 16,

        kGI2DDirtyAll = 0xffffffff
    };

    // This class provides an interface for querying the tracable via geometric operations
    class TracableInterface
    {
    public:
        /*__host__ __device__ virtual bool                    IntersectRay(Ray2D& ray, HitCtx2D& hit, float& tFar) const { return false; }
        __host__ __device__ virtual bool                    InteresectPoint(const vec2& p, const float& thickness) const { return false; }
        __host__ __device__ virtual bool                    IntersectBBox(const BBox2f& bBox) const { return false; }

        __host__ __device__ virtual vec2                    PerpendicularPoint(const vec2& p) const { return vec2(0.0f); }
        __host__ __device__ virtual float                   Evaluate(const vec2& p, const float& thickness, const float& dPdXY) const { return 0.0f; }*/

        __host__ __device__ virtual const BBox2f&           GetBoundingBox() const { return m_tracableBBox; }

    protected:
        __host__ __device__ TracableInterface() {};

        __host__ __device__ __forceinline__ RayBasic2D ToObjectSpace(const Ray2D& world) const
        {
            RayBasic2D obj;
            obj.o = world.o - m_transform.trans;
            obj.d = world.d + obj.o;
            obj.o = m_transform.fwd * obj.o;
            obj.d = (m_transform.fwd * obj.d) - obj.o;
            return obj;
        }

    protected:
        BBox2f                      m_tracableBBox;
        BidirectionalTransform2D    m_transform;
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
                              public Cuda::AssetTags<Host::Tracable, Device::Tracable>
        {
        public:
            __host__ virtual uint       OnCreate(const std::string& stateID, const UIViewCtx& viewCtx) = 0;
            __host__ virtual uint       OnSelect() = 0;
            //__host__ virtual uint       OnSelectElement(const std::string& stateID, const UIViewCtx& viewCtx, UISelectionCtx& selectCtx) = 0;
            __host__ virtual uint       OnMove(const std::string& stateID) = 0;

            __host__ virtual bool       IsEmpty() const = 0;
            __host__ virtual void       Rebuild() = 0;

            __host__ uint               GetDirtyFlags() const { return m_dirtyFlags; }

        protected:
            __host__ Tracable(const std::string& id) : RenderObject(id), m_dirtyFlags(0) {}

            __host__ void               SetDirtyFlags(const uint flags) { m_dirtyFlags |= flags; }
            __host__ void               ClearDirtyFlags(const uint flags) { m_dirtyFlags &= ~flags; }

            uint                        m_dirtyFlags;
        };
    }
}