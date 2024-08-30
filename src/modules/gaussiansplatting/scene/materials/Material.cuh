#pragma once

#include "../SceneObject.cuh"
#include "core/3d/Ctx.cuh"
#include "core/3d/Ray.cuh"
#include "../../scene/SceneContainer.cuh"

namespace Enso
{   
    namespace Device
    {
        // This class provides an interface for querying the tracable via geometric operations
        class Material : public Device::SceneObject
        {
        public:
            __device__ Material();
            __device__ virtual float    Sample(const vec2& xi, const Ray& incident, const HitCtx& hit, vec3& o, vec3& weight) const = 0;
            __device__ virtual float    Evaluate(const Ray& incident, const Ray& extant, const HitCtx& hit, vec3& weight) const = 0;
            __device__ virtual bool     IsPerfectSpecular() const = 0;

            __device__ void             Synchronise(const Device::Vector<Device::Texture2D*>* textures) { m_textures = textures; }

        protected:
            __device__ vec3             EvaluateTexture(const vec2& uv, const int idx) const;

        protected:
            const Device::Vector<Device::Texture2D*>* m_textures;
        };
    }

    namespace Host
    {
        class Material : public Host::SceneObject
        {
        public:
            __host__ Device::Material* GetDeviceInstance() { return cu_deviceInstance; }

        protected:
            __host__ Material(const Asset::InitCtx& initCtx, AssetHandle<Host::SceneContainer>& scene);
            
            __host__ void               SetDeviceInstance(Device::Material* deviceInstance) { cu_deviceInstance = deviceInstance; }
            __host__ void               Bind(AssetHandle<Host::SceneContainer>& scene);

            __host__ virtual void       Synchronise(const uint syncFlags) override final;

        protected:
            __host__ virtual void       OnSynchroniseMaterial(const uint) = 0;

        protected:
            Device::Material*           cu_deviceInstance = nullptr;
            Device::Vector<Device::Texture2D*>* cu_textures;
        };
    }
}