#pragma once

//#include <stdio.h>

namespace Enso
{
    enum GI2DDirtyFlagsDeprecated : unsigned int
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

        // A change to the number of objects in the scene
        kDirtyRebind = 32,

        // Changes that affect integrated values
        kDirtyIntegrators = kDirtyMaterials | kDirtyObjectBounds | kDirtyObjectBVH,

        kDirtyAll = 0xffffffff
    };

    enum GI2DDirtyFlags : unsigned int
    {
        kDirtyObjectBoundingBox            // Called whenever an objects bounding box is updated       
    };
}