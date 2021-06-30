#pragma once

#include "CudaTracable.cuh"

namespace Cuda
{
    namespace Host { class Plane; }

    struct PlaneParams
    {
        __host__ __device__ PlaneParams() : isBounded(false) {}
        __host__ PlaneParams(const ::Json::Node& node, const uint flags) { FromJson(node, flags); }

        __host__ void ToJson(::Json::Node& node) const;
        __host__ void FromJson(const ::Json::Node& node, const uint flags);
        
        bool isBounded;
        BidirectionalTransform transform;
    };

    namespace Device
    {
        class Plane : public Device::Tracable
        {
            friend Host::Plane;
        private:
            PlaneParams m_params;

        public:
            __device__ Plane() {}
            __device__ ~Plane() {}

            __device__ virtual bool Intersect(Ray& ray, HitCtx& hit) const override final;
            __device__ void Synchronise(const PlaneParams& params)
            {
                m_params = params;
            }
        };
    }

    namespace Host
    {
        class Plane : public Host::Tracable
        {
        private:
            Device::Plane* cu_deviceData;
            Device::Plane  m_hostData;

        public:
            __host__ Plane(const ::Json::Node& node);
            __host__ virtual ~Plane() = default;

            __host__ static AssetHandle<Host::RenderObject> Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);
            __host__ static std::string GetAssetTypeString() { return "plane"; }

            __host__ virtual void OnDestroyAsset() override final;
            __host__ virtual void FromJson(const ::Json::Node& node, const uint flags) override final;

            __host__ virtual Device::Plane* GetDeviceInstance() const override final { return cu_deviceData; }
        };
    }
}