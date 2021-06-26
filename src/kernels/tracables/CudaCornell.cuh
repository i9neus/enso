#pragma once

#include "CudaTracable.cuh"

namespace Cuda
{
    namespace Host { class Cornell; }

    namespace Device
    {
        class Cornell : public Device::Tracable
        {
            friend Host::Cornell;
        protected:
            Cornell() = default;

            bool m_isBounded;

        public:
            __device__ Cornell(const BidirectionalTransform& transform) {}
            __device__ ~Cornell() = default;

            __device__ bool Intersect(Ray& ray, HitCtx& hit) const;
        };
    }

    namespace Host
    {
        class Cornell : public Host::Tracable
        {
        private:
            Device::Cornell* cu_deviceData;
            Device::Cornell  m_hostData;

        public:
            __host__ Cornell();
            __host__ virtual ~Cornell() = default;
            __host__ virtual void OnDestroyAsset() override final;

            __host__ static AssetHandle<Host::RenderObject> Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);
            
            __host__ static std::string GetAssetTypeString() { return "cornell"; }
            __host__ virtual Device::Cornell* GetDeviceInstance() const override final { return cu_deviceData; }
        };
    }
}