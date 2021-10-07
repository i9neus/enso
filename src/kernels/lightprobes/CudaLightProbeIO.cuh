#pragma once

#include "CudaLightProbeFilter.cuh"

namespace Json { class Node; }

namespace Cuda
{
    namespace Host { class LightProbeIO; }

    struct LightProbeIOParams
    {
        __host__ __device__ LightProbeIOParams();
        __host__ LightProbeIOParams(const ::Json::Node& node);

        __host__ void ToJson(::Json::Node& node) const;
        __host__ void FromJson(const ::Json::Node& node, const uint flags);

        bool        doExportUSD;
    };

    namespace Host
    {
        class LightProbeIO : public Host::RenderObject
        {
        private:
            AssetHandle<Host::LightProbeGrid>       m_hostInputGrid;
            AssetHandle<Host::LightProbeGrid>       m_hostOutputGrid;

            LightProbeIOParams                      m_params;
            std::string                             m_inputGridID;
            std::string                             m_outputGridID;
            std::string                             m_usdImportPath;
            std::string                             m_usdExportPath;

        public:
            __host__ LightProbeIO(const ::Json::Node& jsonNode, const std::string& id);
            __host__ virtual ~LightProbeIO() = default;

            __host__ static AssetHandle<Host::RenderObject>     Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);

            __host__ virtual void                               FromJson(const ::Json::Node& node, const uint flags) override final;
            __host__ virtual void								Bind(RenderObjectContainer& sceneObjects) override final;
            __host__ virtual void                               ImportProbeGrid();
            __host__ virtual void						        OnPostRenderPass() override final;
            __host__ virtual void                               OnDestroyAsset() override final;
            __host__ virtual void								OnUpdateSceneGraph(RenderObjectContainer& sceneObjects) override final;

            __host__ static std::string                         GetAssetTypeString() { return "probeio"; }
            __host__ static std::string                         GetAssetDescriptionString() { return "Light Probe IO"; }
            __host__ static AssetType                           GetAssetStaticType() { return AssetType::kLightProbeFilter; }
            __host__ virtual std::vector<AssetHandle<Host::RenderObject>> GetChildObjectHandles() override final;
        };
    }
}