#pragma once

#include "core/DirtinessGraph.cuh"

namespace Enso
{
    enum DirtinessFlags : unsigned int
    {
        kClean = 0,

        // Viewport object bounding box has changed
        kDirtyViewportObjectBBox,            

        // Viewport needs redrawing
        kDirtyViewportRedraw,

        // Scene object requires a rebuild
        kDirtyObjectRebuild,
        
        // Scene object is created or destroyed
        kDirtyObjectExistence, 

        // Parameters changed (e.g. through deserialisation) and to be re-synced
        kDirtyParams,

        kNumDirtinessFlags
    };
}