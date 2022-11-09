#pragma once

#include "../tracables/Tracable.cuh"
#include "../FwdDecl.cuh"

using namespace Cuda;

namespace GI2D
{
    class LineSegment;

    struct OmniLightParams
    {
        __host__ __device__ OmniLightParams() {}

        float       m_lightRadius;    
        vec2        m_lightPosWorld;
    };

    namespace Device
    {
        class Light : public Device::Tracable
        {
        public:
            __host__ __device__ Light() {}
        };

        class OmniLight : public Device::Light,
                          public OmniLightParams
        {
        protected:
            __device__ virtual vec4 EvaluatePrimitives(const vec2& pWorld, const UIViewCtx& viewCtx) const override final;

        public:
            __device__ OmniLight() {}
        };
    }

    namespace Host
    {
        class BIH2DAsset;

        class LightInterface
        {
        public:
            __host__ virtual Device::Light* GetDeviceInstance() const = 0;

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