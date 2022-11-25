#include "Tracable.cuh"
#include "core/math/ColourUtils.cuh"

namespace Enso
{
    __host__ __device__ bool Device::Tracable::IntersectBBox(const BBox2f& bBox) const
    {
        return bBox.Intersects(m_objectBBox);
    }
}