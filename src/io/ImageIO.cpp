#include "ImageIO.h"
#include "generic/FilesystemUtils.h"

#include "thirdparty/lodepng/lodepng.h"

namespace ImageIO
{
    void WriteAccumulationBufferPNG(const std::vector<Cuda::vec4>& rawData, const Cuda::ivec2& dimensions, std::string filePath, const float gamma)
    {        
        if (!ReplaceExtension(filePath, "png"))
        {
            Log::Error("Error writing png image: the filename '%s' is invalid.", filePath);
            return;
        }

        std::vector<unsigned char> outData(dimensions.x * dimensions.y * 4, 0);
        uint outIdx = 0;
        const float invGamma = 1.0 / gamma;

        for (int32_t y = 0; y < dimensions.y; ++y)
        {
            for (int32_t x = 0; x < dimensions.x; ++x, outIdx += 4)
            {
                const int inIdx = (dimensions.y - y - 1) * dimensions.x + x;
                Cuda::vec4 pixel = rawData[inIdx];

                // Normalise and saturate the pixel value
                pixel.xyz /= Cuda::max(pixel.w, 1.0f);
                pixel.xyz = Cuda::saturate(Cuda::pow(pixel.xyz, Cuda::vec3(invGamma)));

                for (int32_t c = 0; c < 3; c++)
                {
                    outData[outIdx + c] = uint8_t(pixel[c] * 255.f);
                }
                outData[outIdx + 3] = 255;
            }
        }

        unsigned error = lodepng::encode(filePath,
            outData,
            dimensions.x,
            dimensions.y,
            LodePNGColorType::LCT_RGBA);

        if (error)
        {
            Log::Error("Error writing png image: failed to write image to '%s' (%s)", filePath, lodepng_error_text(error));
        }
        else
        {
            Log::Write("Wrote PNG to '%s'\n", filePath);
        }
    }

}