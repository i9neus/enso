#pragma once

#include "Light.cuh"
#include "../FwdDecl.cuh"
#include "../primitives/Ellipse.cuh"

namespace Enso
{
    class LineSegment;

    struct OmniLightParams
    {
        __host__ __device__ OmniLightParams() : lightRadius(0.0f), lightColour(1.0f), lightIntensity(0.0f) {}
        __device__ void Validate() const {}

        float   lightRadius;

        vec3    lightColour;
        float   lightIntensity;
    };

    namespace Host
    {
        class OmniLight;
    }

    namespace Device
    {
        class OmniLight : public Device::Light
        {  
            friend class Host::OmniLight;

        public:
            __device__ OmniLight() {}

            __host__ __device__ virtual bool            IntersectRay(const Ray2D& ray, HitCtx2D& hit) const override final;
            __host__ __device__ virtual uint            OnMouseClick(const UIViewCtx& viewCtx) const override final;

            __device__ virtual bool                     Sample(const Ray2D& parentRay, const HitCtx2D& hit, float xi, vec2& extant, vec3& L, float& pdf) const override final;
            __device__ virtual bool                     Evaluate(const Ray2D& parentRay, const HitCtx2D& hit, vec3& L, float& pdfLight) const override final;
            __device__ virtual float                    Estimate(const Ray2D& parentRay, const HitCtx2D& hit) const override final;

            __host__ __device__ virtual vec4            EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx) const override final;

            __host__ __device__ virtual void            OnSynchronise(const int) override final;
            __device__ void                             Synchronise(const OmniLightParams& params)
            { 
                m_params = params; 
                OnSynchronise(kSyncParams);
            }

        private:
            OmniLightParams m_params;

            Ellipse m_primitive;
        };
    }

    namespace Host
    {
        class BIH2DAsset;

        class OmniLight : public Host::Light
        {
        public:
            __host__ OmniLight(const std::string& id);
            __host__ virtual ~OmniLight();

            __host__ virtual void       OnDestroyAsset() override final;

            __host__ virtual uint       OnCreate(const std::string& stateID, const UIViewCtx& viewCtx) override final;
            __host__ virtual uint       OnMouseClick(const UIViewCtx& viewCtx) const override final;
            //__host__ virtual uint       OnMove(const std::string& stateID, const UIViewCtx& viewCtx) override final;

            //__host__ virtual uint       OnSelectElement(const std::string& stateID, const vec2& mousePos, const UIViewCtx& viewCtx, UISelectionCtx& selectCtx) override final;
            __host__ virtual bool       Rebuild(const uint parentFlags, const UIViewCtx& viewCtx);

            __host__ static AssetHandle<Host::GenericObject> Instantiate(const std::string& id, const Json::Node&, const AssetHandle<const Host::SceneDescription>&);
            __host__ static const std::string  GetAssetClassStatic() { return "omnilight"; }
            __host__ virtual std::string       GetAssetClass() const override final { return GetAssetClassStatic(); }

            __host__ void               Synchronise(const int syncType);

            __host__ virtual Device::OmniLight* GetDeviceInstance() const override final
            {
                return cu_deviceInstance;
            }

            __host__ virtual bool       Serialise(Json::Node& rootNode, const int flags) const override final;
            __host__ virtual uint       Deserialise(const Json::Node& rootNode, const int flags) override final;

        protected:
            __host__ virtual BBox2f     RecomputeObjectSpaceBoundingBox() override final;

        private:
            Device::OmniLight*          cu_deviceInstance = nullptr;
            Device::OmniLight           m_hostInstance;

            struct
            {
                bool isCentroidSet;
            }
            m_onCreate;
        };
    }
}