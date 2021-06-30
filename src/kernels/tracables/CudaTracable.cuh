#pragma once

#include "../CudaRay.cuh"
#include "../CudaCtx.cuh"
#include "CudaGenericIntersectors.cuh"
#include "../CudaRenderObject.cuh"

namespace Json { class Node; }

namespace Cuda
{
    namespace Host  { class Tracable;  }
    
    namespace Device
    {
        class Material;
        
        class Tracable : public Device::RenderObject, public AssetTags<Device::Tracable, Device::Tracable>
        {
        public:
            __device__ virtual bool Intersect(Ray& ray, HitCtx& hit) const = 0;
            __device__ virtual void InitialiseKernelConstantData() const {};

            __device__ void Synchronise(const Device::Material* material) { cu_material = material; }
            __device__ const Device::Material* GetBoundMaterial() const { return cu_material; }
            
            __device__ virtual ~Tracable() {}

        protected:
            __device__ Tracable() : cu_material(nullptr) {}

            const Device::Material* cu_material;
        };
    }

    namespace Host
    {
        class Material;
        
        class Tracable : public Host::RenderObject, public AssetTags<Host::Tracable, Device::Tracable>
        {
        protected:
            __host__ Tracable() = default;
            __host__ virtual ~Tracable() = default;

            std::string         m_materialId;

        public:
            __host__ virtual Device::Tracable*      GetDeviceInstance() const = 0;
            __host__ virtual AssetType              GetAssetType() const override final { return AssetType::kTracable; }
            __host__ virtual void                   Bind(RenderObjectContainer& objectContainer) override final;
            __host__ virtual void                   FromJson(const ::Json::Node& node, const uint flags) override;
            __host__ static std::string             GetAssetTypeString() { return "tracable"; }
        };
    }
}