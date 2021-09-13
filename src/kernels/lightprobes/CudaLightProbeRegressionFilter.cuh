#pragma once

#include "CudaLightProbeFilter.cuh"

namespace Json { class Node; }

namespace Cuda
{
    namespace Host { class LightProbeRegressionFilter; }

    struct LightProbeRegressionFilterParams
    {
        __host__ __device__ LightProbeRegressionFilterParams();
        __host__ LightProbeRegressionFilterParams(const ::Json::Node& node);

        __host__ void ToJson(::Json::Node& node) const;
        __host__ void FromJson(const ::Json::Node& node, const uint flags);

        bool    isNullFilter;
        int     polynomialOrder;
        int     regressionRadius;
        int     reconstructionRadius;
        int     regressionIterations;
    };

    namespace Host
    {
        class LightProbeRegressionFilter : public Host::RenderObject
        {
        public:
            struct Objects
            {
                LightProbeRegressionFilterParams    params;
                LightProbeFilterGridData            gridData;

                int                                 polyCoeffsPerCoefficient;
                int                                 polyCoeffsPerProbe;
                int                                 numPolyCoeffs;
                int                                 probesPerBatch;     
                
                struct
                {
                    int volume;
                    int span;
                    int radius;
                }
                regression;

                struct
                {
                    int volume;
                    int span;
                    int radius;
                } 
                reconstruction;

                Device::Array<vec3>*                cu_polyCoeffs = nullptr;
                Device::Array<float>*               cu_regressionWeights = nullptr;
            };

        private:
            DeviceObjectRAII<Objects>               m_objects;
            AssetHandle<Host::LightProbeGrid>       m_hostInputGrid;
            AssetHandle<Host::LightProbeGrid>       m_hostInputHalfGrid;
            AssetHandle<Host::LightProbeGrid>       m_hostOutputGrid;
            AssetHandle<Host::Array<vec3>>          m_hostPolyCoeffs;
            AssetHandle<Host::Array<float>>         m_hostRegressionWeights;

            std::string                             m_inputGridID;
            std::string                             m_inputGridHalfID;
            std::string                             m_outputGridID;

            int                                     m_gridSize;
            int                                     m_blockSize;
            int                                     m_probeRange;

        public:
            __host__ LightProbeRegressionFilter(const ::Json::Node& jsonNode, const std::string& id);
            __host__ virtual ~LightProbeRegressionFilter() = default;

            __host__ static AssetHandle<Host::RenderObject>     Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);

            __host__ virtual void                               FromJson(const ::Json::Node& node, const uint flags) override final;
            __host__ virtual void								Bind(RenderObjectContainer& sceneObjects) override final;
            __host__ virtual void                               Prepare();
            __host__ virtual void						        OnPostRenderPass() override final;
            __host__ virtual void                               OnDestroyAsset() override final;
            __host__ virtual void								OnUpdateSceneGraph(RenderObjectContainer& sceneObjects) override final;

            __host__ static std::string                         GetAssetTypeString() { return "proberegressionfilter"; }
            __host__ static std::string                         GetAssetDescriptionString() { return "Light Probe Regression Filter"; }
            __host__ static AssetType                           GetAssetStaticType() { return AssetType::kLightProbeFilter; }
            __host__ virtual std::vector<AssetHandle<Host::RenderObject>> GetChildObjectHandles() override final;
        };
    }
}