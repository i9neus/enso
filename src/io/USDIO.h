#pragma once

#include "generic/StdIncludes.h"
#include "kernels/lightprobes/CudaLightProbeGrid.cuh"

namespace USDIO
{
    enum class SHPackingFormat : int { kNone, kUnity };
    
    void ReadGridDataUSD(std::vector<Cuda::vec3>& gridData, Cuda::LightProbeGridParams& gridParams, const std::string usdImportPath, const SHPackingFormat shFormat);

    void WriteGridDataUSD(const std::vector<Cuda::vec3>& gridData, const Cuda::LightProbeGridParams& gridParams, std::string usdExportPath, const SHPackingFormat shFormat);
    void ExportLightProbeGrid(const Cuda::AssetHandle<Cuda::Host::LightProbeGrid>& grid, const std::string& usdExportPath, const SHPackingFormat shFormat);
}