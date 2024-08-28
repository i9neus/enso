#pragma once

#include <vector>
#include <string>
#include "core/math/Math.cuh"

namespace Enso
{
    namespace ImageIO
    {
        void WriteAccumulationBufferPNG(const std::vector<vec4>& data, const ivec2& dataDimensions, std::string filePath, const float exposure = 1.0f, const float gamma = 2.2f);

        void ReadEXR(std::vector<float>& data, int& width, int& height, int& depth);
    }
}
