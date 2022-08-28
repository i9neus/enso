#pragma once

#include "kernels/lightprobes/CudaLightProbeGrid.cuh"

namespace USDIO
{
    enum class SHPackingFormat : int { kNone, kUnity };
    
    void ReadGridDataUSD(std::vector<Cuda::vec3>& gridData, Cuda::LightProbeGridParams& gridParams, const std::string usdImportPath, const SHPackingFormat shFormat);

    void WriteGridDataUSD(const std::vector<Cuda::vec3>& gridData, const Cuda::LightProbeGridParams& gridParams, std::string usdExportPath, const SHPackingFormat shFormat);
    void WriteGridDataJSON(const std::vector<Cuda::vec3>& gridData, const Cuda::LightProbeGridParams& gridParams, std::string jsonExportPath, const SHPackingFormat shFormat);

    void ExportLightProbeGrid(const Cuda::AssetHandle<Cuda::Host::LightProbeGrid>& grid, Cuda::LightProbeGridParams& exportParams, const std::string& exportPath, const SHPackingFormat shFormat);
}