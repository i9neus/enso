#pragma once

#include "CudaLightProbeGrid.cuh"

namespace Json { class Node; }

namespace Cuda
{
    namespace Host { class LightProbeKernelFilter; }

    enum LightProbeKernelFilterType : int
    {
        kKernelFilterGaussian
    };

    struct LightProbeKernelFilterParams
    {
        __host__ __device__ LightProbeKernelFilterParams() : filterType(kKernelFilterGaussian), radius(1.0f) {}
        __host__ LightProbeKernelFilterParams(const ::Json::Node& node);

        __host__ void ToJson(::Json::Node& node) const;
        __host__ void FromJson(const ::Json::Node& node, const uint flags);

        int filterType;
        float radius;
    };  

    namespace Host
    {
        class LightProbeKernelFilter : public Host::RenderObject
        {
        private:
            LightProbeKernelFilterParams            m_params;
            AssetHandle<Host::LightProbeGrid>       m_grid;
            AssetHandle<Host::Array<vec3>>          m_gridSHData;
            AssetHandle<Host::Array<vec3>>          m_swapBuffer;
                 
        public:
            __host__ LightProbeKernelFilter(const ::Json::Node& jsonNode);
            __host__ virtual ~LightProbeKernelFilter() = default;

            __host__ static AssetHandle<Host::RenderObject>     Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);

            __host__ virtual void                               FromJson(const ::Json::Node& node, const uint flags) override final;
            __host__ virtual void								Bind(RenderObjectContainer& sceneObjects) override final; 
            __host__ virtual void						        OnPostRenderPass() override final;
            __host__ virtual void                               OnDestroyAsset() override final;
            __host__ static std::string                         GetAssetTypeString() { return "probekernelfilter"; }
            __host__ static std::string                         GetAssetDescriptionString() { return "Light Probe Kernel Filter"; }
            __host__ static AssetType                           GetAssetStaticType() { return AssetType::kLightProbeFilter; }
        };
    }
}