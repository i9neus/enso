#pragma once

#include "kernels/math/CudaMath.cuh"

namespace GI2D
{
    enum GI2DDirtyFlags : uint
    {
        kGI2DClean = 0,

        // View params i.e. camera position and orientation
        kGI2DDirtyView = 1,

        // UI changes like selection and lassoing
        kGI2DDirtyUI = 2,

        // Primitive attributes that don't affect the hierarchy like material properteis
        kGI2DDirtyPrimitiveAttributes = 4,

        // Changes to geometry that require a complete rebuild of the hierarchy
        kGI2DDirtyGeometry = 8,

        // Changes to the number of scene objects
        kGI2DDirtySceneObjectCount = 16,

        kGI2DDirtyAll = 0xffffffff
    };
}