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
    class BidirectionalTransform;
    
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
        class Material;
        class Texture2D;
        class DrawableObject;
        class Camera;
        class VoxelProxyGrid;
        class AccumulationBuffer;
        class SceneContainer;

        template<typename> class Vector;
    }

    namespace Host
    {
        class Asset;
        class BIH2DAsset;
        class DrawableObject;
        class Light;
        class Material;
        class Tracable;
        class Texture2D;
        class Camera;
        class VoxelProxyGrid;
        class ComponentContainer;
        class ComponentBuilder;
        class AccumulationBuffer;
        class GenericObjectContainer;
        class PathTracer;    
        class OverlayLayer;
        class SceneContainer;

        template<typename> class Vector;
        template<typename, typename> class AssetVector;
    }
}