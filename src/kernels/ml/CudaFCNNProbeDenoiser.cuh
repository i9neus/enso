#pragma once

#include "../lightprobes/CudaLightProbeGrid.cuh"

#include "onnx/FCNNProbeDenoiser.h"

namespace Json { class Node; }

namespace Cuda
{
    struct FCNNProbeDenoiserParams
    {
        __host__ __device__ FCNNProbeDenoiserParams();
        __host__ FCNNProbeDenoiserParams(const ::Json::Node& node);

        __host__ void ToJson(::Json::Node& node) const;
        __host__ void FromJson(const ::Json::Node& node, const uint flags);       

        Cuda::LightProbeDataTransformParams dataTransform;

        bool            doEvaluate;
    };
    
    namespace Host
    {
        class FCNNProbeDenoiser : public Host::RenderObject
        {
        public:
            __host__ FCNNProbeDenoiser(const std::string& id, const ::Json::Node& jsonNode);
            __host__ virtual ~FCNNProbeDenoiser() = default;

            __host__ void                                       Execute();

            __host__ static AssetHandle<Host::RenderObject>     Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);

            __host__ virtual void                               FromJson(const ::Json::Node& node, const uint flags) override final;
            __host__ virtual void								Bind(RenderObjectContainer& sceneObjects) override final;
            //__host__ virtual void                               Prepare();
            __host__ virtual void                               OnDestroyAsset() override final;
            __host__ virtual void								OnUpdateSceneGraph(RenderObjectContainer& sceneObjects) override final;
            
            __host__ static std::string                         GetAssetTypeString() { return "fcnnprobedenoiser"; }
            __host__ static std::string                         GetAssetDescriptionString() { return "FCNN Probe Denoiser"; }
            __host__ static AssetType                           GetAssetStaticType() { return AssetType::kLightProbeFilter; }
            __host__ virtual std::vector<AssetHandle<Host::RenderObject>> GetChildObjectHandles() override final;

        private:
            __host__ void                                       OnBuildInputGrids(const RenderObject& originObject, const std::string& eventID);

        private:
            AssetHandle<Host::LightProbeGrid>       m_hostInputGrid;
            AssetHandle<Host::LightProbeGrid>       m_hostOutputGrid;
            std::vector<vec3>                       m_rawData;

            std::string                             m_inputGridID;
            std::string                             m_outputGridID;
            
            FCNNProbeDenoiserParams                 m_params;
            ONNX::FCNNProbeDenoiserParams           m_onnxParams;
            ONNX::FCNNProbeDenoiser                 m_onnxEvaluator;

            int                                     m_gridSize;
            int                                     m_blockSize;
            int                                     m_probeRange;
            bool                                    m_isActive;
            bool                                    m_isValidInput;
        };
    }
}