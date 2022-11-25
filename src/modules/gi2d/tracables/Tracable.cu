#include "Tracable.cuh"
#include "kernels/math/CudaColourUtils.cuh"

namespace GI2D
{
    __host__ __device__ bool Device::Tracable::IntersectBBox(const BBox2f& bBox) const
    {
        return bBox.Intersects(m_objectBBox);
    }
}