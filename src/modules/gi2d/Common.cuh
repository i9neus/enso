#pragma once

#include "core/math/Math.cuh"
#include "core/AssetAllocator.cuh"

namespace Enso
{
    enum GI2DDirtyFlags : uint
    {
        kClean = 0,

        // View params i.e. camera position and orientation
        kDirtyView = 1,

        // UI changes like selection and lassoing
        kDirtyUI = 2,

        // Primitive attributes that don't affect the hierarchy like material properteis
        kDirtyMaterials = 4,
        
        // Changes to the boundary of one or more objects (e.g. transform change, BHV change)
        kDirtyObjectBounds = 8,

        // Changes the internal BHV of an object
        kDirtyObjectBVH = 16,

        // Changes that affect integrated values
        kDirtyIntegrators = kDirtyMaterials | kDirtyObjectBounds | kDirtyObjectBVH,

        kDirtyAll = 0xffffffff
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