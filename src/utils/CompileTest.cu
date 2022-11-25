#include "core/math/Math.cuh"

namespace Enso
{
    __device__ void CompileTestDevice()
    {
        int a = max(1, 2);
    }

    __host__ void CompileTestHost()
    {
        int a = max(1, 2);
    }
}