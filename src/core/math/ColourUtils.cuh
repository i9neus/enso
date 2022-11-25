#pragma once

#include "Math.cuh"

namespace Enso
{
    enum ColourSpace : int
    {
        kColourSpaceRGB,
        kColourSpaceCIEXYZ,
        kColourSpaceCIExyY,
        kColourSpaceChroma
    };

    __host__ __device__ vec3 Hue(float phi);

    __host__ __device__ vec3 RGBToChroma(const vec3& rgb);
    __host__ __device__ vec3 ChromaToRGB(const vec3& chro);

    __host__ __device__ vec3 RGBToCIEXYZ(const vec3& rgb);
    __host__ __device__ vec3 CIEXYZToRGB(const vec3& xyz);

    __host__ __device__ vec3 XYZToxyY(const vec3& rgb);
    __host__ __device__ vec3 xyYToXYZ(const vec3& xyy);
    
    __host__ __device__ vec3 HSLToRGB(const vec3& hsl);
    //__host__ __device__ vec3 RGBToHSL(vec3 rgb);

    __host__ __device__ vec3 HSVToRGB(const vec3& hsv);
    __host__ __device__ vec3 RGBToHSV(const vec3& rgb);

    __host__ __device__ vec3 Heatmap(float phi);

    __host__ __device__ vec4 Blend(const vec4& rgba1, const vec4& rgba2);
    __host__ __device__ vec4 Blend(const vec4& rgba1, const vec3& rgb2, const float& w2);

    #define kBlack vec3(0.0f)
    #define kWhite vec3(1.0f)
    #define kRed vec3(1.0f, 0.0f, 0.0f)
    #define kYellow vec3(1.0f, 1.0f, 0.0f)
    #define kGreen vec3(0.0f, 1.0f, 0.0f)
    #define kBlue vec3(0.0f, 0.0f, 1.0f)
    #define kPink vec3(1.0f, 0.0f, 1.0f)
}
