#pragma once

#include "BIHData.cuh"
#include <functional>
#include <vector>
 
namespace Enso
{
    namespace BIH2D
    {
        template<typename NodeType>
        class Builder
        {
        public:
            __host__ Builder(BIHData<NodeType>& bih, Host::Vector<NodeType>& hostNodes, Host::Vector<uint>& hostIndices,
                             const uint minBuildablePrims, std::function<BBox2f(uint)>& getPrimitiveBBox) noexcept;

            __host__ void Build(const bool printStats = false);
            __host__ const Stats& GetStats() const { return m_stats; }

            std::function<void(const char*)> m_debugFunctor = nullptr;


        protected:
            __host__ void BuildPartition(const int i0, const int i1, const uint thisIdx, const uchar depth, const BBox2f& parentBBox, const BBox2f& centroidBBox);

        private:
            BIHData<NodeType>&              m_bih;
            Host::Vector<NodeType>&         m_hostNodes;
            Host::Vector<uint>&             m_hostIndices;
            std::function<BBox2f(uint)>     m_getPrimitiveBBox;
            Stats                           m_stats;
            const uint                      m_minBuildablePrims;
        };

        // Explicitly declare instances of this class for both node types
        template class Builder<FullNode>;
        template class Builder<CompactNode>;
    }
}