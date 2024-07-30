#pragma once

namespace Enso
{
    // BIH
    template<typename> class BIH2D;
    template<typename> class BIH2DNodeBase;
    struct BIH2DNodeDataCompact;
    struct BIH2DNodeDataFull;
    using BIH2DCompactNode = BIH2DNodeBase<BIH2DNodeDataCompact>;
    using BIH2DFullNode = BIH2DNodeBase<BIH2DNodeDataFull>;
    
    template<size_t> class UIButtonMap;
    using VirtualKeyMap = UIButtonMap<256>;

    struct UIViewCtx;    

    template<typename T> class AssetHandle;

    namespace Json
    {
        class Node;
    }

    namespace Generic
    {
        template<typename> class Vector;
    }

    namespace Device
    {
        class BIH2DAsset;
        class Tracable;
        class Light;
        class SceneObject;
        class Camera;
        class VoxelProxyGrid;
        class AccumulationBuffer;
        template<typename> class Vector;
    }

    namespace Host
    {
        class Asset;
        class BIH2DAsset;
        class SceneObject;
        class Light;
        class Tracable;
        class Camera;
        class VoxelProxyGrid;
        class SceneContainer;
        class SceneBuilder;
        class AccumulationBuffer;
        class GenericObjectContainer;
        class PathTracer;

        template<typename> class Vector;
        template<typename, typename> class AssetVector;
    }
}