#pragma once

#include "CudaTracable.cuh"

namespace Cuda
{
    namespace Host { class CornellBox; }

    struct CornellBoxParams
    {
        __host__ __device__ CornellBoxParams() : faceMask(0xffffffu, 6), cameraRayMask(1 << 5, 6) {}
        __host__ CornellBoxParams(const ::Json::Node& node, const uint flags) { FromJson(node, flags); }

        __host__ void ToJson(::Json::Node& node) const;
        __host__ void FromJson(const ::Json::Node& node, const uint flags);

        TracableParams tracable;
        JitterableFlags faceMask;
        JitterableFlags cameraRayMask;
    };

    namespace Device
    {
        class CornellBox : public Device::Tracable
        {
            friend Host::CornellBox;
        private:
            CornellBoxParams m_params;

        public:
            __device__ CornellBox() {}
            __device__ ~CornellBox() {}

            __device__ virtual bool Intersect(Ray& ray, HitCtx& hit) const override final;
            __device__ void Synchronise(const CornellBoxParams& params)
            {
                m_params = params;
            }
        };
    }

    namespace Host
    {
        class CornellBox : public Host::Tracable
        {
        private:
            Device::CornellBox* cu_deviceData;
            CornellBoxParams    m_params;

        public:
            __host__ CornellBox(const std::string& id);
            __host__ CornellBox(const std::string& id, const ::Json::Node& node);
            __host__ virtual ~CornellBox() = default;

            __host__ static AssetHandle<Host::RenderObject> Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);
            __host__ static std::string GetAssetTypeString() { return "cornellbox"; }
            __host__ static std::string GetAssetDescriptionString() { return "Cornell Box"; }
            __host__ virtual int GetIntersectionCostHeuristic() const override final { return 10; };
            __host__ virtual const RenderObjectParams* GetRenderObjectParams() const override final { return &m_params.tracable.renderObject; }

            __host__ virtual void OnDestroyAsset() override final;
            __host__ virtual void FromJson(const ::Json::Node& node, const uint flags) override final;

            __host__ virtual Device::CornellBox* GetDeviceInstance() const override final { return cu_deviceData; }

            __host__ void SetBoundMaterialID(const std::string& id) { m_materialId = id; }
        };
    }
}