﻿#pragma once

#include "CudaLight.cuh"

namespace Cuda
{
    namespace Host 
    { 
        class QuadLight;  
        class Plane;
        class EmitterMaterial;
    }

    struct QuadLightParams : public AssetParams
    {
        __host__ __device__ QuadLightParams() : position(0.0f), orientation(0.0f), scale(1.0f), intensity(1.0f), colour(1.0f) {}
        __host__ QuadLightParams(const ::Json::Node& node);

        __host__ void ToJson(::Json::Node& node) const;
        __host__ void FromJson(const ::Json::Node& node, const uint flags);
        __host__ bool operator==(const QuadLightParams&) const;

        vec3 position;
        vec3 orientation;
        vec3 scale;
        float intensity;
        vec3 colour;

        BidirectionalTransform transform;
    };

    namespace Device
    {        
        class QuadLight : public Device::Light
        {
            friend Host::QuadLight;
        protected:
            float                   m_emitterArea;
            vec3                    m_emitterRadiance;
            QuadLightParams         m_params;

        public:
            __device__ QuadLight();
            __device__ virtual ~QuadLight() {}

            __device__ void Prepare(); 
            __device__ bool Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, vec3& extant, vec3& L, float& pdf) const override final;
            __device__ void Evaluate(const Ray& incident, const HitCtx& hitCtx, vec3& L, float& pdfLight) const override final;
            __device__ void Synchronise(const QuadLightParams& params)
            {  
                m_params = params; 
                Prepare();
            }
        };
    }

    namespace Host
    {
        class QuadLight : public Host::Light
        {
        private:
            Device::QuadLight* cu_deviceData;
            QuadLightParams  m_params;

            AssetHandle<Host::Plane> m_lightPlaneAsset;
            AssetHandle<Host::EmitterMaterial> m_lightMaterialAsset;

        public:
            __host__ QuadLight(const ::Json::Node& node, const std::string& id);
            __host__ virtual ~QuadLight() = default;

            __host__ static AssetHandle<Host::RenderObject> Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);

            __host__ virtual void FromJson(const ::Json::Node& node, const uint flags) override final;
            __host__ virtual void OnDestroyAsset() override final;
            __host__ static std::string GetAssetTypeString() { return "quad"; }
            __host__ virtual Device::QuadLight* GetDeviceInstance() const override final { return cu_deviceData; }
            __host__ virtual std::vector<AssetHandle<Host::RenderObject>> GetChildObjectHandles() override final;
        };
    }
}