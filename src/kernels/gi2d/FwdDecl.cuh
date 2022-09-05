#pragma once

namespace Cuda
{
    template<typename> class VectorInterface;
    
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

    class SceneObjectInterface;
    class TracableInterface;
    
    namespace Device
    {
        class BIH2DAsset;
    }

    namespace Host
    {
        class BIH2DAsset;
        class SceneObject;
        class Tracable;
    }

    using TracableContainer = Cuda::Host::AssetVector<Host::Tracable, TracableInterface>;
    using WidgetContainer = Cuda::Host::AssetVector<Host::SceneObject, SceneObjectInterface>;
}