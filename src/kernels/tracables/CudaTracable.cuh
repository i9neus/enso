#pragma once

#include "../CudaRay.cuh"
#include "../CudaCtx.cuh"
#include "CudaGenericIntersectors.cuh"
#include "../CudaRenderObject.cuh"

namespace Json { class Node; }

namespace Cuda
{
    namespace Host  { class Tracable;  }

    struct TracableParams
    {
        __host__ __device__ TracableParams();
        __host__ __device__ TracableParams(const BidirectionalTransform& transform_) : TracableParams() { transform = transform_;  }
        __host__ TracableParams(const ::Json::Node& node, const uint flags) { FromJson(node, flags); }

        __host__ void ToJson(::Json::Node& node) const;
        __host__ void FromJson(const ::Json::Node& node, const uint flags);

        RenderObjectParams      renderObject;
        BidirectionalTransform  transform;
        bool                    excludeFromBake;
    };
    
    namespace Device
    {
        class Material;
        
        class Tracable : public Device::RenderObject, public AssetTags<Device::Tracable, Device::Tracable>
        {
        public:
            struct Objects
            {
                __host__ __device__ Objects() : cu_material(nullptr), lightId(0xff) {}

                const Device::Material* cu_material;
                uchar lightId;
            };

            __device__ virtual bool Intersect(Ray& ray, HitCtx& hit) const = 0;
            __device__ virtual void InitialiseKernelConstantData() const {}

            __device__ void Synchronise(const Objects& objects) { m_objects = objects; }
            __device__ __forceinline__ const Device::Material* GetBoundMaterial() const { return m_objects.cu_material; }
            __device__ __forceinline__ uchar GetLightID() const { return m_objects.lightId; }
            
            __device__ virtual ~Tracable() {}

        protected:
            Objects m_objects;

            __device__ Tracable() : m_objects() {}
           
        };
    }

    namespace Host
    {
        class Material;
        
        class Tracable : public Host::RenderObject, public AssetTags<Host::Tracable, Device::Tracable>
        {
        protected:
            __host__ Tracable() : m_lightId(kNotALight) {}
            __host__ virtual ~Tracable() = default;
            
            AssetHandle<Host::Material> m_materialAsset;
            std::string                 m_materialId;
            uchar                       m_lightId;

        public:
            __host__ virtual Device::Tracable*      GetDeviceInstance() const = 0;
            __host__ virtual AssetType              GetAssetType() const override final { return AssetType::kTracable; }
            __host__ virtual void                   Bind(RenderObjectContainer& objectContainer) override final;
            __host__ virtual void                   Synchronise() override final;
            __host__ virtual void                   FromJson(const ::Json::Node& node, const uint flags) override;
            __host__ virtual int                    GetIntersectionCostHeuristic() const = 0;
            __host__ static std::string             GetAssetTypeString() { return "tracable"; }
            
            __host__ void                           SetLightID(const uchar lightId) { m_lightId = lightId; }
            __host__ void                           SetBoundMaterialID(const std::string& id) { m_materialId = id; }
        };
    }
}