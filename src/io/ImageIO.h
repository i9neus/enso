#pragma once

#include "generic/StdIncludes.h"
#include "kernels/math/CudaMath.cuh"

namespace ImageIO
{
    void WriteAccumulationBufferPNG(const std::vector<Cuda::vec4>& data, const Cuda::ivec2& dataDimensions, std::string filePath, const float gamma = 2.2f);
}
