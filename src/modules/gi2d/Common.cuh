#pragma once

#include "core/math/Math.cuh"
#include "core/AssetAllocator.cuh"

namespace Enso
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
        
        // Changes to an object transform
        kGI2DDirtyTransforms = 8,

        // Changes to geometry that require a complete rebuild of the hierarchy
        kGI2DDirtyBVH = 16,

        // Changes to the number of scene objects
        kGI2DDirtySceneObjectCount = 32,

        kGI2DDirtyAll = 0xffffffff
    };

    template<typename FlagType>
    __host__ __inline__ bool SetGenericFlags(FlagType& data, const FlagType& newFlags, const bool isSet)
    {
        const FlagType prevData = data;
        if (isSet) { data |= newFlags; }
        else { data &= ~newFlags; }

        return prevData != data;
    }
}