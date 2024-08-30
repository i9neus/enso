#pragma once

#include "../SceneObject.cuh"
#include "core/3d/Ctx.cuh"
#include "core/3d/Ray.cuh"
#include "../../scene/SceneContainer.cuh"
#include "../textures/Texture2D.cuh"
#include "core/containers/Vector.cuh"

namespace Enso
{   
    enum MaterialAttrs : int { kInvalidMaterial = -1 };
    
    namespace Device
    {
        // This class provides an interface for querying the tracable via geometric operations
        class Material : public Device::SceneObject
        {
        public:
            __device__ Material();
            __device__ virtual float    Sample(const vec2& xi, const Ray& incident, const HitCtx& hit, Ray& extant) const = 0;
            __device__ virtual float    Evaluate(const Ray& incident, const Ray& extant, const HitCtx& hit, vec3& weight) const = 0;
            __device__ virtual bool     IsPerfectSpecular() const = 0;

            __device__ void             Synchronise(const Device::Vector<Device::Texture2D*>* textures) { m_textures = textures; }

        protected:
            __device__ __forceinline__ vec3 EvaluateTexture(const vec2& uv, const vec3& def, const int idx) const
            {
                return (idx == -1) ? def : (def * (*m_textures)[idx]->Evaluate(uv).xyz);
            }

            __device__ __forceinline__ float EvaluateTextureLuminance(const vec2& uv, const float def, const int idx) const
            {
                return (idx == -1) ? def : (def * luminance((*m_textures)[idx]->Evaluate(uv).xyz));
            }

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