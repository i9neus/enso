#pragma once

#include "CudaLightProbeGrid.cuh"

namespace Json { class Node; }

namespace Cuda
{
    namespace Host { class LightProbeKernelFilter; }

    enum LightProbeKernelFilterType : int
    {
        kKernelFilterNull,
        kKernelFilterBox,
        kKernelFilterGaussian,
        kKernelFilterNLM
    };

    struct LightProbeKernelFilterParams
    {
        __host__ __device__ LightProbeKernelFilterParams() : filterType(kKernelFilterGaussian), radius(1.0f), trigger(false) {}
        __host__ LightProbeKernelFilterParams(const ::Json::Node& node);

        __host__ void ToJson(::Json::Node& node) const;
        __host__ void FromJson(const ::Json::Node& node, const uint flags);

        int filterType;
        float radius;
        bool trigger;       
    };    

    namespace Host
    {
        class LightProbeKernelFilter : public Host::RenderObject
        {
        public:
            struct Objects
            {                
                LightProbeKernelFilterParams    params;
                ivec3                           gridDensity;
                int                             kernelRadius;
                int                             kernelSpan;
                int                             kernelVolume;
                int                             numProbes;
                int                             blocksPerProbe;
                int                             coefficientsPerProbe;

                const Device::LightProbeGrid*   cu_inputGrid = nullptr;
                Device::LightProbeGrid*         cu_outputGrid = nullptr;
                Device::Array<vec3>*            cu_reduceBuffer = nullptr;
            };

        private:
            Objects                                 m_objects;
            AssetHandle<Host::LightProbeGrid>       m_hostInputGrid;
            AssetHandle<Host::LightProbeGrid>       m_hostOutputGrid;
            AssetHandle<Host::Array<vec3>>          m_hostReduceBuffer;

            std::string                             m_inputGridID;
            std::string                             m_outputGridID;

            int                                     m_gridSize;
            int                                     m_blockSize;
            int                                     m_probeRange;
                 
        public:
            __host__ LightProbeKernelFilter(const ::Json::Node& jsonNode, const std::string& id);
            __host__ virtual ~LightProbeKernelFilter() = default;

            __host__ static AssetHandle<Host::RenderObject>     Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);

            __host__ virtual void                               FromJson(const ::Json::Node& node, const uint flags) override final;
            __host__ virtual void								Bind(RenderObjectContainer& sceneObjects) override final; 
            __host__ virtual void                               Prepare();
            __host__ virtual void						        OnPostRenderPass() override final;
            __host__ virtual void                               OnDestroyAsset() override final;
            __host__ virtual void								OnUpdateSceneGraph(RenderObjectContainer& sceneObjects) override final;

            __host__ static std::string                         GetAssetTypeString() { return "probekernelfilter"; }
            __host__ static std::string                         GetAssetDescriptionString() { return "Light Probe Kernel Filter"; }
            __host__ static AssetType                           GetAssetStaticType() { return AssetType::kLightProbeFilter; }
            __host__ virtual std::vector<AssetHandle<Host::RenderObject>> GetChildObjectHandles() override final;
        };
    }
}