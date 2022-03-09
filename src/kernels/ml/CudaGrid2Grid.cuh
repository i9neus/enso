#pragma once

#include "../lightprobes/CudaLightProbeGrid.cuh"

#include "onnx/Grid2Grid.h"

namespace Json { class Node; }

namespace Cuda
{
    namespace Host { class LightProbeKernelFilter; }

    struct Grid2GridParams
    {
        __host__ __device__ Grid2GridParams();
        __host__ Grid2GridParams(const ::Json::Node& node);

        __host__ void ToJson(::Json::Node& node) const;
        __host__ void FromJson(const ::Json::Node& node, const uint flags);

    };

    namespace Host
    {
        class Grid2Grid : public Host::RenderObject
        {
        public:
            struct Objects
            {
                Grid2GridParams                 params;
                int                             kernelRadius;
                int                             kernelSpan;
                int                             kernelVolume;
                int                             blocksPerProbe;
            };

        private:
            DeviceObjectRAII<Objects>               m_objects;
            AssetHandle<Host::LightProbeGrid>       m_hostInputGrid;
            AssetHandle<Host::LightProbeGrid>       m_hostOutputGrid;
            std::vector<vec3>                       m_rawData;

            std::string                             m_inputGridID;
            std::string                             m_outputGridID;

            ONNX::Grid2Grid                         m_onnxEvaluator;

            int                                     m_gridSize;
            int                                     m_blockSize;
            int                                     m_probeRange;
            bool                                    m_isActive;
            bool                                    m_isValidInput;

        public:
            __host__ Grid2Grid(const std::string& id, const ::Json::Node& jsonNode);
            __host__ virtual ~Grid2Grid() = default;

            __host__ void                                       Execute();

            __host__ static AssetHandle<Host::RenderObject>     Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);

            __host__ virtual void                               FromJson(const ::Json::Node& node, const uint flags) override final;
            __host__ virtual void								Bind(RenderObjectContainer& sceneObjects) override final;
            __host__ virtual void                               Prepare();
            __host__ virtual void                               OnDestroyAsset() override final;
            __host__ virtual void								OnUpdateSceneGraph(RenderObjectContainer& sceneObjects) override final;

            __host__ static std::string                         GetAssetTypeString() { return "grid2grid"; }
            __host__ static std::string                         GetAssetDescriptionString() { return "Grid2Grid"; }
            __host__ static AssetType                           GetAssetStaticType() { return AssetType::kLightProbeFilter; }
            __host__ virtual std::vector<AssetHandle<Host::RenderObject>> GetChildObjectHandles() override final;

        private:
            __host__ void                                       OnBuildInputGrids(const RenderObject& originObject, const std::string& eventID);
        };
    }
}