#include "Camera.cuh"

namespace Enso
{
    __host__ void Host::Camera::Synchronise(const uint syncFlags)
    {
        if (syncFlags & kSyncParams)
        {
            SynchroniseObjects<Device::Camera>(cu_deviceInstance, m_params);
        }
        OnSynchroniseCamera(syncFlags);
    }

    __host__ void Host::Camera::SetPosition(const vec3& cameraPos)
    {
        Prepare(cameraPos, m_params.cameraLookAt, m_params.cameraFov);
        SignalDirty(kDirtySceneObjectChanged);
    }

    __host__ void  Host::Camera::Prepare(const vec3& cameraPos, const vec3& lookAt, const float fov)
    {
        m_params.cameraPos = cameraPos;
        m_params.cameraLookAt = lookAt;
        m_params.cameraFov = fov;
        m_params.inv = CreateBasis(normalize(m_params.cameraPos - m_params.cameraLookAt), vec3(0.f, 1.f, 0.f));
        m_params.fwd = transpose(m_params.inv);
    }
}