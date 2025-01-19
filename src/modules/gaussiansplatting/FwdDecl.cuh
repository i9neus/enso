#pragma once

namespace Enso
{
    namespace BIH2D
    {
        template<typename> class BIHData;
        template<typename> class NodeBase;
        struct NodeDataCompact;
        struct NodeDataFull;
        using CompactNode = NodeBase<NodeDataCompact>;
        using FullNode = NodeBase<NodeDataFull>;
    }

    class BidirectionalTransform;
    class MersenneTwister; 
    class GaussianPoint;
    
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
        class LightSampler;
        class Material;
        class Texture2D;
        class DrawableObject;
        class Camera;
        class VoxelProxyGrid;
        class AccumulationBuffer;
        class SceneContainer;
        class SceneBuilder;
        class GenericObject;
        class GaussianPointCloud;

        template<typename> class Vector;
    }

    namespace Host
    {
        class Asset;
        class BIH2DAsset;
        class DrawableObject;
        class LightSampler;
        class Material;
        class Tracable;
        class Texture2D;
        class Camera;
        class VoxelProxyGrid;
        class AccumulationBuffer;
        class GenericObjectContainer;
        class PathTracer;    
        class ViewportRenderer;
        class SceneContainer;
        class SceneBuilder;
        class GenericObject;
        class RenderableObject;
        class GaussianPointCloud;
        class SceneObject;

        template<typename> class Vector;
        template<typename, typename> class AssetVector;
    }
}