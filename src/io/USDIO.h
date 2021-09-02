#pragma once

#include "generic/StdIncludes.h"
#include "kernels/lightprobes/CudaLightProbeGrid.cuh"

namespace USDIO
{
    void ExportLightProbeGrid(const Cuda::AssetHandle<Cuda::Host::LightProbeGrid>& grid, const std::string& usdExportPath);
}