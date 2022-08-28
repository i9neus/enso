#pragma once

#include <vector>
#include <string>
#include "kernels/math/CudaMath.cuh"

namespace ImageIO
{
    void WriteAccumulationBufferPNG(const std::vector<Cuda::vec4>& data, const Cuda::ivec2& dataDimensions, std::string filePath, const float exposure = 1.0f, const float gamma = 2.2f);
}
