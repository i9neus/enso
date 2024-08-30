#pragma once

#include "Material.cuh"

namespace Enso
{
    struct DiffuseParams
    {
        __host__ __device__ DiffuseParams() : albedo(kOne * 0.5f), albedoTextureIdx(-1) {}
        __host__ __device__ DiffuseParams(const vec3& alb, const int albIdx) : albedo(alb), albedoTextureIdx(albIdx) {}
        __device__ void Validate() const {}

        vec3 albedo;
        int albedoTextureIdx = -1;
    };

    namespace Device
    {
        class Diffuse : public Device::Material
        {
        public:
            __device__                  Diffuse() {}

            __device__ virtual float    Sample(const vec2& xi, const Ray& incident, const HitCtx& hit, Ray& extant) const override final;
            __device__ virtual float    Evaluate(const Ray& incident, const Ray& extant, const HitCtx& hit, vec3& weight) const override final;
            __device__ virtual bool     IsPerfectSpecular() const { return false; }
            __device__ void             Synchronise(const DiffuseParams& params) { m_params = params; }

        private:
            DiffuseParams m_params;
        };
    }

    namespace Host
    {
        class Diffuse : public Host::Material
        {
        public:
            __host__                    Diffuse(const Asset::InitCtx& initCtx, AssetHandle<Host::SceneContainer>& scene, const DiffuseParams& params);
            __host__ virtual            ~Diffuse() noexcept;

        protected:
            __host__ virtual void       OnSynchroniseMaterial(const uint) override final;
            
        protected:
            Device::Diffuse*            cu_deviceInstance = nullptr;

            DiffuseParams               m_params;
        };
    }
}