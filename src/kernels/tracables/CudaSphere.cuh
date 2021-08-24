#pragma once

#include "CudaTracable.cuh"

namespace Json { class Node; }

namespace Cuda
{
    class RenderObjectContainer;
    namespace Host {  class Sphere;   }

    namespace Device
    {
        class Sphere : public Device::Tracable
        {
            friend Host::Sphere;
        protected:
            TracableParams m_params;

        public:
            __device__ Sphere() {}
            __device__ virtual ~Sphere() {}

            __device__ virtual bool Intersect(Ray& ray, HitCtx& hit) const override final; 
            __device__ void Synchronise(const TracableParams& params)
            {
                m_params = params;
            }
        };
    }

    namespace Host
    {        
        class Sphere : public Host::Tracable
        {
        private:
            Device::Sphere* cu_deviceData;
            TracableParams  m_params;

        public:
            __host__ Sphere(const uint flags);
            __host__ Sphere(const ::Json::Node& json);
            __host__ virtual ~Sphere() = default;
            __host__ virtual void OnDestroyAsset() override final;
            
            __host__ static AssetHandle<Host::RenderObject> Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);

            __host__ virtual Device::Sphere* GetDeviceInstance() const override final { return cu_deviceData; }
            __host__ static std::string GetAssetTypeString() { return "sphere"; }
            __host__ static std::string GetAssetDescriptionString() { return "Sphere"; }
            __host__ virtual int GetIntersectionCostHeuristic() const override final { return 10; };
            
            __host__ virtual void FromJson(const ::Json::Node& node, const uint flags) override final;
            __host__ void UpdateParams(const BidirectionalTransform& transform);
        };
    }
}