#pragma once

#include "GlossyConductor.cuh"
#include "core/math/Mappings.cuh"
#include "core/3d/Basis.cuh"
#include "core/3d/bxdfs/GGX.cuh"
#include "../textures/Texture2D.cuh"
#include "core/containers/Vector.cuh"
#include "core/math/ColourUtils.cuh"

namespace Enso
{
    __device__ __forceinline__ float Device::GlossyConductor::EvaluateAlpha(const vec2& uv) const
    {
        if (m_params.roughnessTextureIdx == -1)
        {
            return m_params.roughness;
        }
        else
        {
            const float lum = saturatef(EvaluateTextureLuminance(uv, 1.0f, m_params.roughnessTextureIdx));
            return mix(m_params.roughnessRange, lum);
        }
    }

    __device__ float Device::GlossyConductor::Sample(const vec2& xi, const Ray& incident, const HitCtx& hit, Ray& extant) const
    {
        float ggxWeight;
        const float pdf = BxDF::SampleMicrofacetReflectorGGX(xi, -incident.od.d, hit.n, EvaluateAlpha(hit.uv), extant.od.d, ggxWeight);
        if (pdf > 0)
        {
            extant.od.o = incident.Point() + hit.n * 1e-4;
            extant.weight = EvaluateTexture(hit.uv, m_params.albedo, m_params.albedoTextureIdx) * ggxWeight;
            return pdf;
        }
        return 0;
    }

    __device__ float Device::GlossyConductor::Evaluate(const Ray& incident, const Ray& extant, const HitCtx& hit, vec3& weight) const
    {
        weight = EvaluateTexture(hit.uv, m_params.albedo, m_params.albedoTextureIdx);

        return BxDF::EvaluateMicrofacetReflectorGGX(-incident.od.d, extant.od.o, hit.n, EvaluateAlpha(hit.uv));
    }

    __host__ Host::GlossyConductor::GlossyConductor(const Asset::InitCtx& initCtx, AssetHandle<Host::SceneContainer>& scene, const GlossyConductorParams& params) :
        Material(initCtx, scene),
        m_params(params),
        cu_deviceInstance(AssetAllocator::InstantiateOnDevice<Device::GlossyConductor>(*this))
    {
        Material::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::Material>(cu_deviceInstance));

        Synchronise(kSyncParams | kSyncObjects);
    }

    __host__ Host::GlossyConductor::~GlossyConductor() noexcept
    {
        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);
    }

    __host__ void Host::GlossyConductor::OnSynchroniseMaterial(const uint syncFlags)
    {
        if (syncFlags & kSyncParams)
        {
            SynchroniseObjects<Device::GlossyConductor>(cu_deviceInstance, m_params);
        }
    }

}