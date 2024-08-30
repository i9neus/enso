#pragma once

#include "SpecularDielectric.cuh"
#include "core/math/Mappings.cuh"
#include "core/3d/Basis.cuh"
#include "core/3d/bxdfs/Specular.cuh"
#include "../textures/Texture2D.cuh"
#include "core/containers/Vector.cuh"
#include "core/math/ColourUtils.cuh"

namespace Enso
{
    __device__ float Device::SpecularDielectric::Sample(const vec2& xi, const Ray& incident, const HitCtx& hit, Ray& extant) const
    {
        const bool isSubsurface = BxDF::SamplePerfectDielectric(xi.y, -incident.od.d, hit.n, m_params.ior, extant.od.d, extant.od.o);

        extant.flags = kRaySubsurface * uint(isSubsurface);
        extant.od.o += incident.Point();        

        // If this is a subsurface ray, attenuate it based upon its length
        if (incident.IsSubsurface())
        {
            extant.weight *= exp(-incident.tNear * m_params.colour * m_params.absorption);
        }

        return BxDF::kMaxPDF;

    }

    __device__ float Device::SpecularDielectric::Evaluate(const Ray& incident, const Ray& extant, const HitCtx& hit, vec3& weight) const
    {
        CudaAssertMsg(false, "SpecularDielectric: Cannot evaluate because BxDF is a delta function.");
        return 0;
    }

    __host__ Host::SpecularDielectric::SpecularDielectric(const Asset::InitCtx& initCtx, AssetHandle<Host::SceneContainer>& scene, const SpecularDielectricParams& params) :
        Material(initCtx, scene),
        m_params(params),
        cu_deviceInstance(AssetAllocator::InstantiateOnDevice<Device::SpecularDielectric>(*this))
    {
        Material::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::Material>(cu_deviceInstance));

        m_params.ior = clamp(m_params.ior, 1.f, 10.f);

        Synchronise(kSyncParams | kSyncObjects);
    }

    __host__ Host::SpecularDielectric::~SpecularDielectric() noexcept
    {
        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);
    }

    __host__ void Host::SpecularDielectric::OnSynchroniseMaterial(const uint syncFlags)
    {
        if (syncFlags & kSyncParams)
        {
            SynchroniseObjects<Device::SpecularDielectric>(cu_deviceInstance, m_params);
        }
    }

}