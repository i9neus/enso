#pragma once

#include "Material.cuh"

namespace Enso
{
    struct GlossyConductorParams
    {
        __host__ __device__ GlossyConductorParams() :
            roughnessRange(0.01f, 1.), roughness(0.2f), albedo(0.5f), albedoTextureIdx(-1), roughnessTextureIdx(-1) {}
        __host__ __device__ GlossyConductorParams(const vec3& alb, const float rough, const vec2& range = vec2(), const int albIdx = -1, const int roughIdx = -1) :
            albedo(alb), roughness(rough), roughnessRange(range), albedoTextureIdx(albIdx), roughnessTextureIdx(roughIdx) {}

        __device__ void Validate() const {}

        vec3 albedo;
        float roughness;
        vec2 roughnessRange;
        int albedoTextureIdx;
        int roughnessTextureIdx;
    };

    namespace Device
    {
        class GlossyConductor : public Device::Material
        {
        public:
            __device__                  GlossyConductor() {}

            __device__ virtual float    Sample(const vec2& xi, const Ray& incident, const HitCtx& hit, Ray& extant) const override final;
            __device__ virtual float    Evaluate(const Ray& incident, const Ray& extant, const HitCtx& hit, vec3& weight) const override final;
            __device__ virtual bool     IsPerfectSpecular() const { return false; }
            __device__ void             Synchronise(const GlossyConductorParams& params) { m_params = params; }

        private:
            __device__ __forceinline__ float EvaluateAlpha(const vec2& uv) const;

        private:
            GlossyConductorParams m_params;
        };
    }

    namespace Host
    {
        class GlossyConductor : public Host::Material
        {
        public:
            __host__                    GlossyConductor(const Asset::InitCtx& initCtx, AssetHandle<Host::SceneContainer>& scene, const GlossyConductorParams& params);
            __host__ virtual            ~GlossyConductor() noexcept;

        protected:
            __host__ virtual void       OnSynchroniseMaterial(const uint) override final;

        protected:
            Device::GlossyConductor*    cu_deviceInstance = nullptr;

            GlossyConductorParams       m_params;
        };
    }
}