#pragma once

#include "Material.cuh"

namespace Enso
{
    struct DiffuseParams
    {
        __device__ void Validate() const {}

        int albedoIdx = -1;
    };

    namespace Device
    {
        class Diffuse : public Device::Material
        {
        public:
            __device__ Diffuse();
            __device__ virtual float    Sample(const vec2& xi, const Ray& incident, const HitCtx& hit, vec3& o, vec3& weight) const override final;
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
            __host__ Diffuse(const Asset::InitCtx& initCtx, AssetHandle<Host::SceneContainer>& scene, const int albedoIdx);
            __host__ virtual ~Diffuse() noexcept;

        protected:
            __host__ virtual void       OnSynchroniseMaterial(const uint) override final;
            
        protected:
            Device::Diffuse* cu_deviceInstance = nullptr;

            DiffuseParams     m_params;
        };
    }
}