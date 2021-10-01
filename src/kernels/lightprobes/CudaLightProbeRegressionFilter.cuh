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

        int     filterType;
        int     polynomialOrder;
        int     regressionRadius;
        int     reconstructionRadius;
        int     regressionIterations;
        float   learningRate;
        int     minSamples;
        float   tikhonovCoeff;

        LightProbeFilterNLMParams nlm;
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
                
                int                                 regrCoeffsPerSHCoeff;
                int                                 regrCoeffsPerProbe;
                int                                 totalRegrCoeffs;
                int                                 probesPerBatch;     
                
                struct
                {
                    int volume;
                    int span;
                    int radius;
                    int numMonomials;
                }
                regression;

                struct
                {
                    int volume;
                    int span;
                    int radius;
                } 
                reconstruction;

                Device::Array<vec3>*                cu_C = nullptr;
                Device::Array<float>*               cu_D = nullptr;
                Device::Array<vec3>*                cu_dLdC = nullptr;
                Device::Array<float>*               cu_W = nullptr;
                float*                              cu_T = nullptr;
            };

        private:
            DeviceObjectRAII<Objects>               m_objects;
            AssetHandle<Host::LightProbeGrid>       m_hostInputGrid;
            AssetHandle<Host::LightProbeGrid>       m_hostCrossGrid;
            AssetHandle<Host::LightProbeGrid>       m_hostCrossHalfGrid;
            AssetHandle<Host::LightProbeGrid>       m_hostOutputGrid;
            AssetHandle<Host::Array<vec3>>          m_hostC;
            AssetHandle<Host::Array<float>>         m_hostD;
            AssetHandle<Host::Array<vec3>>          m_hostdLdC;
            AssetHandle<Host::Array<float>>         m_hostW;
            DeviceObjectRAII<float, 4*4*4>          m_hostT;

            std::string                             m_inputGridID;
            std::string                             m_crossGridID;
            std::string                             m_crossGridHalfID;
            std::string                             m_outputGridID;

            struct
            {
                int                                     gridSize;
                int                                     blockSize;
                int                                     sharedMemoryBytes;
            }
            m_regressionKernel;

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

        private:
            __host__ void                                       PrecomputeMonomialMatrices();
        };
    }
}