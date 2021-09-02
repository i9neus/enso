#pragma once

#include "CudaTracable.cuh"

namespace Cuda
{
    namespace Host { class Box; }

    namespace Device
    {
        class Box : public Device::Tracable
        {
            friend Host::Box;
        private:
            TracableParams m_params;

        public:
            __device__ Box() {}
            __device__ ~Box() {}

            __device__ virtual bool Intersect(Ray& ray, HitCtx& hit) const override final;
            __device__ void Synchronise(const TracableParams& params)
            {
                m_params = params;
            }
        };
    }

    namespace Host
    {
        class Box : public Host::Tracable
        {
        private:
            Device::Box* cu_deviceData;
            TracableParams    m_params;

        public:
            __host__ Box();
            __host__ Box(const ::Json::Node& node);
            __host__ virtual ~Box() = default;

            __host__ static AssetHandle<Host::RenderObject> Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);
            __host__ static std::string GetAssetTypeString() { return "box"; }
            __host__ static std::string GetAssetDescriptionString() { return "Box"; }
            __host__ virtual int GetIntersectionCostHeuristic() const override final { return 10; };
            __host__ virtual const RenderObjectParams* GetRenderObjectParams() const override final { return &m_params.renderObject; }

            __host__ virtual void OnDestroyAsset() override final;
            __host__ virtual void FromJson(const ::Json::Node& node, const uint flags) override final;

            __host__ virtual Device::Box* GetDeviceInstance() const override final { return cu_deviceData; }

            __host__ void SetBoundMaterialID(const std::string& id) { m_materialId = id; }
        };
    }
}