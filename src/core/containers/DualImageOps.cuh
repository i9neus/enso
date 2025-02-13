#pragma once

#include "DualImage.cuh"

namespace Enso
{   
    enum : int { kImageDilateSquare, kImageDilateCircle };
    enum : int { kImageRMask = 1, kImageGMask = 2, kImageBMask = 4, kImageAMask = 8 };
    
    template<typename Type, int Channels> void Dilate(Host::DualImage<Type, Channels>& inputImage, const int type, float radius, cudaStream_t stream, const uint channelMask = 0xffffffff);
}