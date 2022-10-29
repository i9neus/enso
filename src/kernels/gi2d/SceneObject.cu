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
}