#pragma once

#include <vector>
#include <string>
#include "core/math/Math.cuh"

namespace Enso
{
    namespace ImageIO
    {
        void WritePNG(const std::vector<vec4>& data, const ivec2& dataDimensions, std::string filePath, const float exposure, const float gamma);
    }
}
