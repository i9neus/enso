#pragma once

#include "CudaMath.cuh"

namespace Cuda
{
    __host__ __device__ vec3 Hue(float phi);
    
    __host__ __device__ vec3 HSLToRGB(const vec3& hsl);
    //__host__ __device__ vec3 RGBToHSL(vec3 rgb);

    __host__ __device__ vec3 HSVToRGB(const vec3& hsv);
    __host__ __device__ vec3 RGBToHSV(const vec3& rgb);

}