#include "Camera.cuh"
#include "core/math/ColourUtils.cuh"
#include "io/json/JsonUtils.h"
#include "core/Vector.cuh"
#include "../Dirtiness.cuh"

namespace Enso
{
    __device__ Device::Camera::Camera() : m_frameIdx(1) {}

    __device__ void Device::Camera::Synchronise(const Device::SceneDescription& scene)
    {
        m_voxelTracer.Synchronise(scene);
    }

    __device__ void Device::Camera::Prepare(const uint dirtyFlags)
    {
        m_frameIdx = (dirtyFlags & kDirtyIntegrators) ? 0 : (m_frameIdx + 1);
    }

    __device__ void Device::Camera::Integrate(const uchar ctxFlags)
    {
        RenderCtx renderCtx(kKernelIdx, uint(m_frameIdx), 0, *this, ctxFlags);

        /*if (ctxFlags & kRenderCtxDebug)
        {
            renderCtx.debugData = &m_kifsDebug;
        }*/

        m_voxelTracer.Integrate(renderCtx);
    }
}