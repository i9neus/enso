#pragma once

#include "Material.cuh"

namespace Enso
{
    struct SpecularDielectricParams
    {
        __host__ __device__ SpecularDielectricParams() :
            ior(1.3f), absorption(1.f), colour(kOne) {}
        __host__ __device__ SpecularDielectricParams(const float i, const float ab, const vec3& col) :
            ior(i), absorption(ab), colour(col) {}

        __device__ void Validate() const {}

        float ior;
        float absorption;
        vec3 colour;
    };

    namespace Device
    {
        class SpecularDielectric : public Device::Material
        {
        public:
            __device__                  SpecularDielectric() {}

            __device__ virtual float    Sample(const vec2& xi, const Ray& incident, const HitCtx& hit, Ray& extant) const override final;
            __device__ virtual float    Evaluate(const Ray& incident, const Ray& extant, const HitCtx& hit, vec3& weight) const override final;
            __device__ virtual bool     IsPerfectSpecular() const { return true; }
            __device__ void             Synchronise(const SpecularDielectricParams& params) { m_params = params; }

        private:
            __device__ __forceinline__ float EvaluateAlpha(const vec2& uv) const;

        private:
            SpecularDielectricParams m_params;
        };
    }

    namespace Host
    {
        class SpecularDielectric : public Host::Material
        {
        public:
            __host__                    SpecularDielectric(const Asset::InitCtx& initCtx, AssetHandle<Host::SceneContainer>& scene, const SpecularDielectricParams& params);
            __host__ virtual            ~SpecularDielectric() noexcept;

        protected:
            __host__ virtual void       OnSynchroniseMaterial(const uint) override final;

        protected:
            Device::SpecularDielectric*    cu_deviceInstance = nullptr;

            SpecularDielectricParams       m_params;
        };
    }
}