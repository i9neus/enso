#include "CudaCompositor.h"

namespace Cuda
{
    __global__ void sinewave_gen_kernel(unsigned int width, unsigned int height, cudaSurfaceObject_t cuSurface, float time)
    {
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) { return; }

        float4 pixel;
        pixel.x = float(x) / float(width - 1);
        pixel.y = float(y) / float(height - 1); 
        pixel.z = 0.0;
        pixel.w = 1.0;
       
        surf2Dwrite(pixel, cuSurface, x * 16, y);
    }

    // The host CPU Sinewave thread spawner
    void CompositeBuffers(unsigned int width, unsigned int height, cudaSurfaceObject_t cuSurface, float time, cudaStream_t streamToRun)
    {
        dim3 block(16, 16, 1);
        dim3 grid(width / 16, height / 16, 1);

        sinewave_gen_kernel << < grid, block, 0, streamToRun >> > (width, height, cuSurface, time);

        getLastCudaError("sinewave_gen_kernel execution failed.\n");
    }

}