#include "CudaCompositor.cuh"

namespace Cuda
{    
    __global__ void KernelSignalSetRead(unsigned int* signal)   { atomicCAS(signal, 0u, 1u); }
    __global__ void KernelSignalUnsetRead(unsigned int* signal) { atomicCAS(signal, 1u, 0u); }
    
    __global__ void KernelCopyImageToD3DTexture(unsigned int width, unsigned int height, const Image* image, cudaSurfaceObject_t cuSurface, unsigned int* signal)
    {
        if (*signal != 1u) { return; }

        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height || x >= image->Width() || y >= image->Height()) { return; }

        float4 pixel;
        pixel.x = float(x) / float(width - 1);
        pixel.y = float(y) / float(height - 1);
        pixel.z = 0.0;
        pixel.w = 1.0;

        surf2Dwrite(pixel, cuSurface, x * 16, y);
    }

    // The host CPU Sinewave thread spawner
    void CopyImageToD3DTexture(unsigned int width, unsigned int height, const Image* image, cudaSurfaceObject_t cuSurface, cudaStream_t hostStream, unsigned int* signal)
    {
        dim3 block(16, 16, 1);
        dim3 grid(width / 16, height / 16, 1); 

        KernelSignalSetRead << < 1, 1, 0, hostStream >> > (signal);
        KernelCopyImageToD3DTexture << < grid, block, 0, hostStream >> > (width, height, image, cuSurface, signal);
        KernelSignalUnsetRead << < 1, 1, 0, hostStream >> > (signal);

        getLastCudaError("sinewave_gen_kernel execution failed.\n");
    }

}