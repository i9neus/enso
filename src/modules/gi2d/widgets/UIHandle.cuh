#pragma once

#include "core/2d/primitives/Ellipse.cuh"
#include "../FwdDecl.cuh"

namespace Enso
{
    namespace Host { template<typename T> class Vector; }

    class UIHandle
    {
    private:


    public:
        enum kControlState : unsigned char { kDeselected = 0, kDragging = 1 };

        __host__ __device__ UIHandle() : m_state(kDeselected) {}
        __host__ __device__ UIHandle(const vec2& o, const float& r) : 
            m_state(kDeselected),
            m_ellipse(o, r) {}

        __host__ __device__ vec4                    EvaluateOverlay(const vec2& p, const UIViewCtx& viewCtx) const;
        __host__ __device__ __forceinline__ bool    Contains(const vec2& p, const UIViewCtx& ctx) const { return EvaluateOverlay(p, ctx).w > 0.f; }
        __host__ __device__ __forceinline__ BBox2f  GetBoundingBox() const { return m_ellipse.GetBoundingBox(); }
        __host__ __device__ __forceinline__ vec2    GetCentroid() const { return m_ellipse.GetOrigin(); }

        __host__ uint                               OnDelegateAction(const std::string& stateID, const VirtualKeyMap& keyMap, const vec2& mousePosObject);

    private:
        int         m_state;
        Ellipse     m_ellipse;
    };
}