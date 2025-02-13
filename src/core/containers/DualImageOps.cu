#include "DualImageOps.cuh"

namespace Enso
{
    template<typename Type, int Channels>
    __global__ void KernelDilateSquare(Device::DualImage<Type, Channels>* destImage, const Device::DualImage<Type, Channels>* srcImage, const ivec2 dir, const int radius, const uint channelMask)
    {
        CudaAssert(destImage && srcImage);
        const int x = kKernelX, y = kKernelY;
        const int width = destImage->Width(), height = destImage->Height();
        if (x < width && y < height)
        {
            Type maxL[Channels];
            memset(maxL, 0, sizeof(Type) * Channels);

            for (int k = -radius; k <= radius; ++k)
            {
                const int i = x + k * dir.x, j = y + k * dir.y;
                if(i >= 0 && j >= 0 && i < width && j < height)
                {
                    const Type* pixel = srcImage->At(i, j);
                    for (int c = 0; c < Channels; ++c)
                    {
                        if (k == 0 || ((1 << c) & channelMask) != 0)
                        {
                            maxL[c] = fmaxf(maxL[c], pixel[c]);
                        }
                    }
                }
            }

            destImage->Set(x, y, maxL);
        }
    }
    
    template<typename Type, int Channels> void Dilate(Host::DualImage<Type, Channels>& inputImage, const int type, float radius, cudaStream_t stream, const uint channelMask)
    {
        AssetHandle<Host::DualImage<Type, Channels>> swapImage = AssetAllocator::CreateChildAsset<Host::DualImage<Type, Channels>>(inputImage, "dilateSwap", inputImage.Width(), inputImage.Height());

        auto [gridDims, blockDims] = Get2DLaunchParams(inputImage.Width(), inputImage.Height());
        if (type == kImageDilateSquare)
        {
            KernelDilateSquare << <gridDims, blockDims, 0, stream >> > (swapImage->GetDeviceInstance(), inputImage.GetDeviceInstance(), ivec2(1, 0), radius, channelMask);
            KernelDilateSquare << <gridDims, blockDims, 0, stream >> > (inputImage.GetDeviceInstance(), swapImage->GetDeviceInstance(), ivec2(0, 1), radius, channelMask);
            IsOk(cudaStreamSynchronize(stream));
        }

        swapImage.DestroyAsset();
    }

    template void Dilate(Host::DualImage<float, 3>& inputImage, const int type, float radius, cudaStream_t stream, const uint channelMask);

    /* template<typename Type, int Channels>
    DualImage<Type, Channels>Downsample(const DualImage<Type, Channels>& inputImg, const int factor)
    {
        DualImage<Type, Channels> newDualImage(inputImg.Width() / factor, inputImg.Height() / factor);

        for (int y = 0, outIdx = 0; y < newDualImage.Height(); ++y)
        {
            for (int x = 0; x < newDualImage.Width(); ++x, outIdx += Channels)
            {
                const int u0 = x * inputImg.Width() / newDualImage.Width();
                const int u1 = (x + 1) * inputImg.Width() / newDualImage.Width();
                const int v0 = y * inputImg.Height() / newDualImage.Height();
                const int v1 = (y + 1) * inputImg.Height() / newDualImage.Height();
                int sumPixels = 0;

                float sigma[Channels] = {};
                for (int v = v0; v < v1; ++v)
                {
                    for (int u = u0; u < u1; ++u)
                    {
                        if (u >= 0 && u < inputImg.Width() && v >= 0 && v < inputImg.Height())
                        {
                            for (int c = 0; c < Channels; ++c)
                            {
                                sigma[c] += inputImg[(v * inputImg.Width() + u) * Channels + c];
                            }
                            ++sumPixels;
                        }
                    }

                    for (int c = 0; c < Channels; ++c) { newDualImage[outIdx + c] = sigma[c] / sumPixels; }
                }
            }
        }

        return newDualImage;
    }

    template<typename Type, int Channels>
    DualImage<Type, Channels> Crop(const DualImage<Type, Channels>& inputImg, DualImageRect cropRegion)
    {
        cropRegion = Intersection(inputImg.Rect(), cropRegion);

        DualImage<Type, Channels> newDualImage(cropRegion.Width(), cropRegion.Height());
        newDualImage.ParallelMap([&](const int x, const int y, const int i, float* outputPixel)
            {
                const float* inputPixel = inputImg.At(x + cropRegion.x0, y + cropRegion.y0);
        for (int c = 0; c < Channels; ++c)
        {
            outputPixel[c] = inputPixel[c];
        }
            });

        return newDualImage;
    }*/
}