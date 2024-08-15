#pragma once

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
        kDirtyObjectRequestRebuild,

        // Scene object requesting objects rebind their assets
        kDirtyObjectRebind,
        
        // Scene object is created or destroyed
        kDirtyObjectExistence, 

        // Parameters changed (e.g. through deserialisation) and to be re-synced
        kDirtyParams,

        // A render object has changed. This should clear accumulators, resync device parameters, etc.
        kDirtySceneObjectChanged,

        kNumDirtinessFlags
    };
}