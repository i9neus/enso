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
    enum TracableFlags : uint
    {
        kTracableSelected = 1u
    };
    
    struct TracableParams
    {
        __host__ __device__ TracableParams() {}

        BBox2f                      objectBBox;
        BBox2f                      worldBBox;

        BidirectionalTransform2D    transform;
        uint                        attrFlags;
    };
    
    // This class provides an interface for querying the tracable via geometric operations
    class TracableInterface
    {
    public:
        __host__ __device__ virtual bool                    IntersectRay(Ray2D& ray, HitCtx2D& hit) const { return false; }
        //__host__ __device__ virtual bool                    InteresectPoint(const vec2& p, const float& thickness) const { return false; }
        __host__ __device__ virtual bool                    IntersectBBox(const BBox2f& bBox) const;

        //__host__ __device__ virtual vec2                    PerpendicularPoint(const vec2& p) const { return vec2(0.0f); }

        __device__ vec4                                     EvaluateOverlay(const vec2& p, const UIViewCtx& viewCtx) const;

        __host__ __device__ const BBox2f&                   GetObjectSpaceBoundingBox() const { return m_params.objectBBox; };
        __host__ __device__ const BBox2f&                   GetWorldSpaceBoundingBox() const { return m_params.worldBBox; };
        __host__ __device__ virtual const vec3              GetColour() const { return kOne; }

        __device__ void                                     Synchronise(const TracableParams& params);      

    protected:
        __host__ __device__ TracableInterface() {}

        __device__ virtual vec4                             EvaluatePrimitives(const vec2& pWorld, const UIViewCtx& viewCtx) const { return vec4(0.0f); }

        __device__ bool                                     EvaluateControlHandles(const vec2& pWorld, const UIViewCtx& viewCtx, vec4& L) const;

        __host__ __device__ __forceinline__ RayBasic2D ToObjectSpace(const Ray2D& world) const
        {
            RayBasic2D obj;
            obj.o = world.o - m_params.transform.trans;
            obj.d = world.d + obj.o;
            obj.o = m_params.transform.fwd * obj.o;
            obj.d = (m_params.transform.fwd * obj.d) - obj.o;
            return obj;
        }

    protected:
        TracableParams m_params;

    private:
        BBox2f m_handleInnerBBox;
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
            //__host__ virtual uint       OnSelectElement(const std::string& stateID, const UIViewCtx& viewCtx, UISelectionCtx& selectCtx) = 0;
            __host__ virtual uint       OnMove(const std::string& stateID, const UIViewCtx& viewCtx);            
            __host__ virtual uint       OnSelect(const bool isSelected);

            __host__ virtual bool       Finalise() = 0;

            __host__ virtual bool       IsEmpty() const = 0;
            __host__ virtual void       Rebuild(const uint parentFlags, const UIViewCtx& viewCtx) = 0;

            __host__ uint               GetDirtyFlags() const { return m_dirtyFlags; }
            __host__ bool               IsFinalised() const { return m_isFinalised; }
            __host__ bool               IsSelected() const { return m_params.attrFlags & kTracableSelected; }

            __host__ 

            __host__ virtual TracableInterface* GetDeviceInstance() const = 0;

            __host__ virtual void SetAttributeFlags(const uint flags, bool isSet = true)
            {
                if (SetGenericFlags(m_params.attrFlags, flags, isSet))
                {
                    SetDirtyFlags(kGI2DDirtyUI, true);
                }
            }

        protected:
            __host__ Tracable(const std::string& id) : RenderObject(id), m_dirtyFlags(0), m_isFinalised(false) {}

            __host__ void SetDirtyFlags(const uint flags, const bool isSet = true) { SetGenericFlags(m_dirtyFlags, flags, isSet); }
            __host__ void ClearDirtyFlags() { m_dirtyFlags = 0; }

            uint                        m_dirtyFlags;
            bool                        m_isFinalised;

            struct
            {
                vec2                        dragAnchor;
                bool                        isDragging;
            }
            m_onMove;
        };
    }
}