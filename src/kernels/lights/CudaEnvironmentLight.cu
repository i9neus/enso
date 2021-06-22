#include "CudaEnvironmentLight.cuh"

namespace Cuda
{
    __device__ Device::EnvironmentLight::EnvironmentLight(const BidirectionalTransform& transform) : Light(transform)
    {
       
    }

    __device__ bool Device::EnvironmentLight::Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, RayBasic& extant, vec3& L, float& pdfLight) const
    {
       
    }

    __device__ void Device::EnvironmentLight::Evaluate()
    {
    }

    __host__  Host::EnvironmentLight::EnvironmentLight()
        : cu_deviceData(nullptr)
    {
        m_hostData.m_transform.MakeIdentity();

        cu_deviceData = InstantiateOnDevice<Device::EnvironmentLight>(m_hostData.m_transform);
    }

    __host__ void Host::EnvironmentLight::OnDestroyAsset()
    {
        DestroyOnDevice(&cu_deviceData);
    }
}