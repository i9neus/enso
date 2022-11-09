#pragma once

namespace Core
{
    template<typename> class Vector;

    namespace Device
    {
        template<typename> class Vector;
    }

    namespace Host
    {
        template<typename> class Vector;
        template<typename, typename> class AssetVector;
    }
}

namespace GI2D
{
    // BIH
    template<typename> class BIH2D;
    template<typename> class BIH2DNodeBase;
    struct BIH2DNodeDataCompact;
    struct BIH2DNodeDataFull;
    using BIH2DCompactNode = BIH2DNodeBase<BIH2DNodeDataCompact>;
    using BIH2DFullNode = BIH2DNodeBase<BIH2DNodeDataFull>;
    
    namespace Device
    {
        class BIH2DAsset;
        class Tracable;
        class Light;
        class SceneObject;
        class Camera2D;
        class VoxelProxyGrid;
    }

    namespace Host
    {
        class BIH2DAsset;
        class SceneObjectInterface;
        class LightInterface;
        class TracableInterface; 
        class VoxelProxyGrid;
        class SceneDescription;

        class OverlayLayer;
        class PathTracerLayer;
        class VoxelProxyGridLayer;
    }
}