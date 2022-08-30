#ifdef FWD_DECL_VECTOR
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
        template<typename> class AssetVector;
    }
}
#endif

#ifdef FWD_DECL_BIH2D
namespace GI2D
{
    template<typename> class BIH2D;
    template<typename> class BIH2DNodeBase;
    struct BIH2DNodeDataCompact;
    struct BIH2DNodeDataFull;

    using BIH2DCompactNode = BIH2DNodeBase<BIH2DNodeDataCompact>;
    using BIH2DFullNode = BIH2DNodeBase<BIH2DNodeDataFull>;
    
    namespace Device
    {
        class BIH2DAsset;
    }

    namespace Host
    {
        class BIH2DAsset;
    }
}
#endif