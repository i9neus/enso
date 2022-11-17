#pragma once

#include "../tracables/Tracable.cuh"
#include "../FwdDecl.cuh"
#include "../tracables/primitives/Ellipse.cuh"

using namespace Cuda;

namespace GI2D
{
    class LineSegment;

    struct OmniLightParams
    {
        __host__ __device__ OmniLightParams() {}

        vec2    m_lightPos;
        float   m_lightRadius;
    };

    namespace Device
    {
        class Light : public Device::Tracable
        {
        public:
            __host__ __device__ Light() {}

            __device__ virtual bool                     Sample(const Ray2D& parentRay, const HitCtx2D& hit, float xi, vec2& extant, vec3& L, float& pdf) const = 0;
            __device__ virtual bool                     Evaluate(const Ray2D& parentRay, const HitCtx2D& hit, vec3& L, float& pdfLight) const = 0;
        };

        class OmniLight : public Device::Light,
                          public OmniLightParams
        {  
        public:
            __device__ OmniLight() {}

            __host__ __device__ virtual bool            IntersectRay(const Ray2D& ray, HitCtx2D& hit) const override final;

            __device__ virtual bool                     Sample(const Ray2D& parentRay, const HitCtx2D& hit, float xi, vec2& extant, vec3& L, float& pdf) const override final;
            __device__ virtual bool                     Evaluate(const Ray2D& parentRay, const HitCtx2D& hit, vec3& L, float& pdfLight) const override final;

            __device__ virtual vec4                     EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx) const override final;

            __device__ virtual void                     OnSynchronise(const int) override final;

        private:
            Ellipse m_primitive;
        };
    }

    namespace Host
    {
        class BIH2DAsset;

        class LightInterface
        {
        public:
            __host__ virtual Device::Light*             GetDeviceInstance() const = 0;

        protected:
            LightInterface() = default;
        };

        template<typename DeviceType>
        class Light : public Host::Tracable<DeviceType>,
                      public LightInterface
        {
            using Super = Host::Tracable<DeviceType>;

        public:
            __host__ Light(const std::string& id) : Super(id) {}
            __host__ virtual ~Light() {}         

        protected:
            template<typename SubType> __host__ inline void Synchronise(SubType* deviceData, const int syncType) { Super::Synchronise(deviceData, syncType); }
        };

        class OmniLight : public Host::Light<Device::OmniLight>
        {
            using Super = Host::Light<Device::OmniLight>;

        public:
            __host__ OmniLight(const std::string& id);
            __host__ virtual ~OmniLight();

            __host__ virtual void       OnDestroyAsset() override final;

            __host__ virtual uint       OnCreate(const std::string& stateID, const UIViewCtx& viewCtx) override final;
            __host__ virtual uint       OnMove(const std::string& stateID, const UIViewCtx& viewCtx) override final;

            //__host__ virtual uint       OnSelectElement(const std::string& stateID, const vec2& mousePos, const UIViewCtx& viewCtx, UISelectionCtx& selectCtx) override final;
            __host__ virtual bool       IsConstructed() const override final;
            __host__ virtual bool       Rebuild(const uint parentFlags, const UIViewCtx& viewCtx) override final;
            __host__ virtual bool       Finalise() override final;

            __host__ static AssetHandle<GI2D::Host::SceneObjectInterface> Instantiate(const std::string& id);
            __host__ static const std::string  GetAssetTypeString() { return "omnilight"; }

            __host__ void               Synchronise(const int syncType);

            __host__ virtual Device::OmniLight* GetDeviceInstance() const override final
            {
                return cu_deviceInstance;
            }

        private:
            Device::OmniLight*          cu_deviceInstance = nullptr;

            struct
            {
                bool isCentroidSet;
            }
            m_onCreate;
        };
    }
}