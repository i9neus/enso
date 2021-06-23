#pragma once

#include "CudaLight.cuh"

namespace Cuda
{
    namespace Host { class QuadLight;  }

    struct QuadLightParams
    {
        __host__ __device__ QuadLightParams() : position(0.0f), orientation(0.0f), scale(1.0f), intensity(1.0f), colour(1.0f) {}
        __host__ QuadLightParams(const Json::Node& node) { FromJson(node); }

        __host__ void ToJson(Json::Node& node) const;
        __host__ void FromJson(const Json::Node& node);

        vec3 position;
        vec3 orientation;
        vec3 scale;

        float intensity;
        vec3 colour;
    };

    namespace Device
    {        
        class QuadLight : public Device::Light
        {
            friend Host::QuadLight;
        protected:
            QuadLight() = default;

            float m_emitterArea;
            vec3  m_emitterRadiance;
            QuadLightParams m_params;

        public:
            __device__ QuadLight(const BidirectionalTransform& transform);
            __device__ ~QuadLight() = default;

            __device__ void Prepare(); 
            __device__ bool Sample(const Ray& incident, const HitCtx& hitCtx, RenderCtx& renderCtx, vec3& extant, vec3& L, float& pdf) const;
            __device__ void Evaluate(const Ray& incident, const HitCtx& hitCtx, vec3& L, float& pdfLight) const;
            __device__ void OnSyncParameters(const QuadLightParams& params) 
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
            Device::QuadLight  m_hostData;

        public:
            __host__ QuadLight();
            __host__ virtual ~QuadLight() { OnDestroyAsset(); }
            __host__ virtual void OnJson(const Json::Node& jsonNode) override final;
            __host__ virtual void OnDestroyAsset() override final;

            __host__ virtual Device::QuadLight* GetDeviceInstance() const override final { return cu_deviceData; }
        };
    }
}