#include "UIHandle.cuh"
#include "core/Vector.cuh"
#include "core/math/Math.cuh"
#include "core/VirtualKeyStates.h"
#include "core/UIButtonMap.h"

#include <random>

namespace Enso
{
    __host__ __device__ vec4 UIHandle::EvaluateOverlay(const vec2& p, const UIViewCtx& viewCtx) const
    {
        return m_ellipse.EvaluateOverlay(p, OverlayCtx(viewCtx).SetFill(vec4(kOne * 0.2, 1.f)).SetStroke(vec4(1.f), 2.f));
    }
    
    __host__ uint UIHandle::OnDelegateAction(const std::string& stateID, const VirtualKeyMap& keyMap, const vec2& mousePosObject)
    {
        if (m_state == kDeselected && keyMap.IsSet(VK_LBUTTON) && m_ellipse.Contains(mousePosObject, 0.f))
        {
            m_state = kDragging;
            return kDirtyObjectBounds;
        }
        else if (m_state == kDragging)
        {
            if (!keyMap.IsSet(VK_LBUTTON))
            {
                m_state = kDeselected;
                return kDirtyObjectBounds;
            }
            else
            {
                m_ellipse.SetOrigin(mousePosObject);
                return kDirtyObjectBounds;
            }
        }

        return 0u;
    }
}