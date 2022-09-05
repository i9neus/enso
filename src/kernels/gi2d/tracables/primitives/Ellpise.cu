#include "Ellipse.cuh"
#include "kernels/CudaVector.cuh"

#include <random>

using namespace Cuda;

namespace GI2D
{
    __host__ __device__ float Ellipse::Evaluate(const vec2& p, const float& dPdXY) const
    {
        //return Cuda::saturate(1.0f - (length(p - PerpendicularPoint(p)) - thickness) / dPdXY);
        return 0.0f;
    }    
}