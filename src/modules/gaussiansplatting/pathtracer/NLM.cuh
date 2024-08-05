#pragma once

#include "core/Image.cuh"

namespace Enso
{
    class NLMDenoiser
    {
    public:
        __device__ NLMDenoiser();

        __device__ void Initialise(const int N, const int M, const float alpha, const float K);
        __device__ void Initialise(Device::ImageRGBW* meanAccumBuffer, Device::ImageRGBW* varAccumBuffer);

        __device__ __forceinline__ bool IsValidTexel(const ivec2& p);
        __device__ __forceinline__ bool GetTexel(const ivec2& p, vec3& P, vec3& varP);
        __device__ __forceinline__ float PatchDistance(const ivec2& p, const ivec2& q);
        
        __device__ vec3 FilterPixel(const ivec2& p);
        __device__ vec3 FilterPixelBox(const ivec2& p);

    private:
        int m_kM;
        int m_kN;
        float m_kAlpha;
        float m_kK;

        ivec2 m_viewportDims;

        Device::ImageRGBW* m_meanAccumBuffer = nullptr;
        Device::ImageRGBW* m_varAccumBuffer = nullptr;
    };
}