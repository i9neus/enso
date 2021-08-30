#pragma once

#include "../CudaRay.cuh"
#include "../CudaCtx.cuh"
#include "../CudaRenderObject.cuh"

namespace Cuda
{
    namespace Host 
    { 
        class Light; 
        class Tracable;
    }

    struct LightParams
    {
        __host__ __device__ LightParams();
        __host__ LightParams(const ::Json::Node& node);

        __host__ void ToJson(::Json::Node& node) const;
        __host__ void FromJson(const ::Json::Node& node, const uint flags);

        JitterableFloat   intensity;
        JitterableVec3    colourHSV;

        RenderObjectParams renderObject;
        BidirectionalTransform transform;
    };

    namespace Device
    {
        class Light : public Device::RenderObject, public AssetTags<Host::Light, Device::Light>
        {
        public:
            struct Objects
            {
                uchar               lightId;
            }
            m_objects;

            __device__ Light() {}
            __device__ virtual ~Light() {}

            __device__ virtual bool Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, vec2 xi, vec3& extant, vec3& L, float& pdf) const = 0;
            __device__ virtual bool Evaluate(const Ray& incident, const HitCtx& hitCtx, vec3& L, float& pdfLight) const = 0;
            __device__ virtual float Estimate(const Ray& incident, const HitCtx& hitCtx) const = 0;
            __device__ virtual uchar GetLightRayFlags() const { return 0; }

            __device__ void Synchronise(const Objects& objects) { m_objects = objects; }
        };
    }

    namespace Host
    {
        class Light : public Host::RenderObject, public AssetTags<Host::Light, Device::Light>
        {
        protected:
            uchar                   m_lightId;

        public:
            __host__ virtual void FromJson(const ::Json::Node& node, const uint flags) override
            {
                Host::RenderObject::UpdateDAGPath(node);
            }
            __host__ virtual Device::Light*                 GetDeviceInstance() const = 0;
            __host__ virtual AssetHandle<Host::Tracable>    GetTracableHandle();
            __host__ virtual void                           Synchronise() override final;

            __host__ virtual AssetType                      GetAssetType() const override final { return AssetType::kLight; }
            __host__ static AssetType                       GetAssetStaticType() { return AssetType::kLight; }
            __host__ static std::string                     GetAssetTypeString() { return "light"; }
            __host__ static uint                            GetInstanceFlags() { return kInstanceFlagsAllowMultipleInstances; }

            __host__ uchar                                  GetLightID() const { return m_lightId; }
            __host__ void                                   SetLightID(const uchar id) { m_lightId = id; }
        };
    }
}