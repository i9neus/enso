#pragma once

//#include <stdio.h>

namespace Enso
{
    enum GI2DDirtyFlags : unsigned int
    {
        // Scene object bounding box has changed
        kDirtyObjectBoundingBox,            

        // Scene object requires a rebuild
        kDirtyObjectRebuild,
        
        // Scene object is created or destroyed
        kDirtyObjectExistence,              
        
        // Contents of integrators is invalid,
        kDirtyIntegrators,

        // Overlay elements are invalid and need redrawing
        kDirtyUIOverlay,

        // Parameters changed (e.g. through deserialisation) and to be re-synced
        kDirtyParams
    };
}