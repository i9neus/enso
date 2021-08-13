#pragma once

#include "generic/StdIncludes.h"
#include "kernels/CudaLightProbeGrid.cuh"

namespace USDIO
{
    void TestUSD();

    void ExportLightProbeGrid(const Cuda::AssetHandle<Cuda::Host::LightProbeGrid>& grid);
}