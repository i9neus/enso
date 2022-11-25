#include "SceneObject.cuh"
#include "kernels/math/CudaColourUtils.cuh"

namespace GI2D
{
    __device__ bool Device::SceneObject::EvaluateControlHandles(const vec2& pWorld, const UIViewCtx& viewCtx, vec4& L) const
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

    __host__ Host::SceneObject::SceneObject(const std::string& id, Device::SceneObject& hostInstance) :
        RenderObject(id),
        m_dirtyFlags(kGI2DDirtyAll),
        m_isFinalised(false),
        m_hostInstance(hostInstance)
    {
    }

    __host__ uint Host::SceneObject::OnSelect(const bool isSelected)
    {
        if (SetGenericFlags(m_attrFlags, uint(kSceneObjectSelected), isSelected))
        {
            SetDirtyFlags(kGI2DDirtyUI, true);
        }
        return m_dirtyFlags;
    }

    __host__ uint Host::SceneObject::OnMove(const std::string& stateID, const UIViewCtx& viewCtx)
    {
        if (stateID == "kMoveSceneObjectBegin")
        {
            m_onMove.dragAnchor = viewCtx.mousePos;
            m_onMove.isDragging = true;
            Log::Error("kMoveSceneObjectBegin");
        }
        else if (stateID == "kMoveSceneObjectDragging")
        {
            Assert(m_onMove.isDragging);

            const vec2 delta = viewCtx.mousePos - m_onMove.dragAnchor;
            m_onMove.dragAnchor = viewCtx.mousePos;
            m_transform.trans += delta;
            m_worldBBox += delta;

            // The geometry internal to this object hasn't changed, but it will affect the 
            Log::Warning("kMoveSceneObjectDragging");
            SetDirtyFlags(kGI2DDirtyBVH);
        }
        else if (stateID == "kMoveSceneObjectEnd")
        {
            m_onMove.isDragging = false;
            Log::Success("kMoveSceneObjectEnd");
        }

        return m_dirtyFlags;
    }
}