#include "Tracable.cuh"
#include "kernels/math/CudaColourUtils.cuh"

namespace GI2D
{
    __device__ vec4 TracableInterface::EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx) const
    {
        vec4 L = EvaluatePrimitives(pWorld, viewCtx);

        if (m_params.attrFlags & kTracableSelected)
        {
            EvaluateControlHandles(pWorld, viewCtx, L);
        }

        return L;
    }

    __host__ __device__ bool TracableInterface::IntersectBBox(const BBox2f& bBox) const
    {
        return bBox.Intersects(m_params.objectBBox);
    }
    
    __device__ bool TracableInterface::EvaluateControlHandles(const vec2& pWorld, const UIViewCtx& viewCtx, vec4& L) const
    {
        // Draw the bounding box
        /*if (m_params.objectBBox.PointOnPerimiter(p, m_params.viewCtx.dPdXY * 2.f))
        {
            L = vec4(kOne, 1.0f);
            return true;
        }*/

        // Draw the control handles
        if (m_handleInnerBBox.IsValid() && !m_handleInnerBBox.Contains(pWorld))
        {
            //L = Blend(L, kOne, 0.2f);

            /*vec2 hp = 2.0f * (p - m_handleInnerBBox.lower) / (m_handleInnerBBox.Dimensions() - vec2(m_params.viewCtx.dPdXY * 10.0f));
             
            if (fract(hp.x) < 0.1f && fract(hp.y) < 0.1f) 
            { 
                 L = vec4(1.0f); 
                 return true;
            }*/
        }        

        return false;
    }

    __device__ void TracableInterface::Synchronise(const TracableParams& params)
    {
        m_params = params;

        //m_handleInnerBBox = Grow(m_params.objectBBox, m_params.viewCtx.dPdXY * -5.0f);        
    }

    __host__ uint Host::Tracable::OnSelect(const bool isSelected)
    {
        if (SetGenericFlags(m_params.attrFlags, uint(kTracableSelected), isSelected))
        {
            SetDirtyFlags(kGI2DDirtyUI, true);
        }
        return m_dirtyFlags;
    }

    __host__ uint Host::Tracable::OnMove(const std::string& stateID, const UIViewCtx& viewCtx)
    {
        if (stateID == "kMoveTracableBegin")
        {
            m_onMove.dragAnchor = viewCtx.mousePos;
            m_onMove.isDragging = true;
            Log::Error("kMoveTracableBegin");
        }
        else if (stateID == "kMoveTracableDragging")
        {
            Assert(m_onMove.isDragging);

            const vec2 delta = viewCtx.mousePos - m_onMove.dragAnchor;
            m_onMove.dragAnchor = viewCtx.mousePos;
            m_params.transform.trans += delta;
            m_params.worldBBox += delta;

            // The geometry internal to this object hasn't changed, but it will affect the 
            Log::Warning("kMoveTracableDragging");
            SetDirtyFlags(kGI2DDirtyTransforms);
        }
        else if (stateID == "kMoveTracableEnd")
        {
            m_onMove.isDragging = false;
            Log::Success("kMoveTracableEnd");
        }

        return m_dirtyFlags;
    }
}