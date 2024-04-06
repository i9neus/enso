#pragma once

#include "../Dirtiness.cuh"
#include "core/Image.cuh"
#include "../Ray2D.cuh"
#include "../UICtx.cuh"

namespace Enso
{
    namespace Host { template<typename T> class Vector; }

    enum GI2DPrimitiveFlags : int
    {
        k2DPrimitiveSelected = 1
    };

    #define kPrimitiveColourNone vec4(0.f)

    struct OverlayCtx
    {
        enum __flags : unsigned char { kOverlayStroke = 1, kOverlayFill = 2 };
        
        __host__ __device__ OverlayCtx(const UIViewCtx& viewCtx) :
            flags(0),
            dPdXY(viewCtx.dPdXY),
            fillColour(0.f),
            strokeColour(0.f),
            strokeThickness(1e-10f) {} 

        __host__ __device__ static OverlayCtx MakeStroke(const UIViewCtx& viewCtx, const vec4& colour, const float& thickness)
        {
            OverlayCtx ctx(viewCtx);
            return ctx.SetStroke(colour, thickness);
        }

        __host__ __device__ static OverlayCtx MakeFill(const UIViewCtx& viewCtx, const vec4& colour)
        {
            OverlayCtx ctx(viewCtx);
            return ctx.SetFill(colour);
        }

        __host__ __device__ OverlayCtx& SetStroke(const vec4& colour, const float& thickness) { strokeColour = colour; strokeThickness = thickness; flags |= kOverlayStroke; return *this; }
        __host__ __device__ OverlayCtx& SetFill(const vec4& colour) { fillColour = colour; flags |= kOverlayFill; return *this; }

        __host__ __device__ __forceinline__ bool HasStroke() const { return flags & kOverlayStroke; }
        __host__ __device__ __forceinline__ bool HasFill() const { return flags & kOverlayFill; }

        unsigned char   flags;
        float           dPdXY;
        vec4            fillColour;
        vec4            strokeColour;
        float           strokeThickness;
    };

    /*class Primitive2D
    {
    protected:
        __host__ __device__ Primitive2D() noexcept : m_flags(0) {}
        __host__ __device__ Primitive2D(const uchar& f) noexcept : m_flags(f) {}

        uchar m_flags;

    public:
        __host__ __device__ virtual vec2                    PerpendicularPoint(const vec2& p) const { dassert("PerpendicularPoint not implemented."); return vec2(0.0f); }
        __host__ __device__ virtual vec4                    EvaluateOverlay(const vec2& p, const OverlayCtx& ctx) const { dassert("EvaluateOverlay not implemented."); return vec4(0.0f); }
        __host__ __device__ virtual bool                    TestPoint(const vec2& p, const float& thickness) const { dassert("TestPoint not implemented."); return false; }
        __host__ __device__ virtual bool                    IntersectRay(const RayBasic2D& ray, HitCtx2D& hit) const { dassert("IntersectRay not implemented."); return kFltMax; }
        __host__ __device__ virtual BBox2f                  GetBoundingBox() const { dassert("GetBoundingBox not implemented."); return BBox2f::MakeInvalid(); }
        __host__ __device__ virtual bool                    Intersects(const BBox2f& bBox) const { dassert("Intersects not implemented."); return false; }
        __host__ __device__ virtual bool                    Contains(const vec2& p, const float& dPdXY) const { dassert("Contains not implemented."); return false; }

        __host__ __device__ __forceinline__ bool            IsSelected() const { return m_flags & k2DPrimitiveSelected; }
        __host__ __device__ __forceinline__ void            SetFlags(const uchar flags, const bool set)
        {
            if (set) { m_flags |= flags; }
            else { m_flags &= ~flags; }
        }

        //__host__ __device__ __forceinline__ void            UnsetFlags(const uchar flags) { m_flags &= ~flags; }
    };*/
}