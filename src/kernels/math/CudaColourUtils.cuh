#pragma once

#include "CudaMath.cuh"

namespace Cuda
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

    __host__ __device__ vec4 Blend(vec4 lowerRgba, const vec4& upperRgba);
    __host__ __device__ vec4 Blend(vec4 lowerRgba, const vec3& upperRgb, const float& upperAlpha);
}
