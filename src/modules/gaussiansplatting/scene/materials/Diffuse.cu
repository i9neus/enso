#pragma once

#include "Diffuse.cuh"
#include "core/math/Mappings.cuh"
#include "core/3d/Basis.cuh"
#include "core/3d/bxdfs/Lambert.cuh"
#include "../textures/Texture2D.cuh"
#include "core/containers/Vector.cuh"
#include "core/math/ColourUtils.cuh"

namespace Enso
{
    __device__ Device::Diffuse::Diffuse()
    {
    }

    __device__ float Device::Diffuse::Sample(const vec2& xi, const Ray& incident, const HitCtx& hit, vec3& o, vec3& weight) const
    {
        weight = EvaluateTexture(hit.uv, m_params.albedoIdx);

        return BxDF::SampleLambertian(xi, hit.n, o);
    }

    __device__ float Device::Diffuse::Evaluate(const Ray& incident, const Ray& extant, const HitCtx& hit, vec3& weight) const
    {
        weight = EvaluateTexture(hit.uv, m_params.albedoIdx);

        return BxDF::EvaluateLambertian();
    }

    __host__ Host::Diffuse::Diffuse(const Asset::InitCtx& initCtx, AssetHandle<Host::SceneContainer>& scene, const int albedoIdx) :
        Material(initCtx, scene),
        cu_deviceInstance(AssetAllocator::InstantiateOnDevice<Device::Diffuse>(*this))
    {
        Material::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::Material>(cu_deviceInstance));

        m_params.albedoIdx = albedoIdx;
        Synchronise(kSyncParams | kSyncObjects);
    }

    __host__ Host::Diffuse::~Diffuse() noexcept
    {
        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);
    }

    __host__ void Host::Diffuse::OnSynchroniseMaterial(const uint syncFlags)
    {
        if (syncFlags & kSyncParams)
        {
            SynchroniseObjects<Device::Diffuse>(cu_deviceInstance, m_params);
        }
    }

}