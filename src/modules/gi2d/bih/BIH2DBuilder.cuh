#pragma once

#include "BIH2D.cuh"
#include <functional>
#include <vector>
 
namespace Enso
{
    template<typename NodeType>
    class BIH2DBuilder
    {
    public:
        __host__ BIH2DBuilder(BIH2D<NodeType>& bih, Host::Vector<NodeType>& hostNodes, Host::Vector<uint>& hostIndices,
            const uint minBuildablePrims, std::function<BBox2f(uint)>& getPrimitiveBBox) noexcept;

        __host__ void Build(const bool printStats = false);

        std::function<void(const char*)> m_debugFunctor = nullptr;


    protected:
        __host__ void BuildPartition(const int i0, const int i1, const uint thisIdx, const uchar depth, const BBox2f& parentBBox, const BBox2f& centroidBBox);

    private:
        BIH2D<NodeType>&                            m_bih;
        Host::Vector<NodeType>&                     m_hostNodes;
        Host::Vector<uint>&                         m_hostIndices;
        std::function<BBox2f(uint)>                 m_getPrimitiveBBox;
        BIH2DStats&                                 m_stats;
        const uint                                  m_minBuildablePrims;
    };

    // Explicitly declare instances of this class for both node types
    template class BIH2DBuilder<BIH2DFullNode>;
    template class BIH2DBuilder<BIH2DCompactNode>;
}