#include "Builder.cuh"
#include "core/utils/HighResolutionTimer.h"

namespace Enso
{
    namespace BIH2D
    {
        template<typename NodeType>
        __host__ Builder<NodeType>::Builder(BIHData<NodeType>& bih, Host::Vector<NodeType>& hostNodes, Host::Vector<uint>& hostIndices,
            const uint minBuildablePrims, std::function<BBox2f(uint)>& functor) noexcept :
            m_bih(bih),
            m_hostNodes(hostNodes),
            m_hostIndices(hostIndices),
            m_getPrimitiveBBox(functor),
            m_minBuildablePrims(minBuildablePrims)
        {
        }

        template<typename NodeType>
        __host__ void Builder<NodeType>::Build(const bool printStats)
        {
            HighResolutionTimer timer;
            AssertMsg(m_getPrimitiveBBox, "BIH builder does not have a valid bounding box functor.");

            // Find the global bounding box
            BBox2f centroidBBox = BBox2f::Invalid();
            m_bih.bBox = BBox2f::Invalid();

            if (!m_hostIndices.empty())
            {
                for (const auto idx : m_hostIndices)
                {
                    const BBox2f primBBox = m_getPrimitiveBBox(idx);
                    //AssertMsgFmt(primBBox.IsValid(),
                    //    "BIH primitive at index %i has returned an invalid bounding box: {%s, %s}", idx, primBBox[0].format().c_str(), primBBox[1].format().c_str());
                    if (primBBox.IsValid())
                    {
                        m_bih.bBox = Union(m_bih.bBox, primBBox);
                        centroidBBox = Union(centroidBBox, primBBox.Centroid());
                    }
                }

                //AssertMsgFmt(m_bih.bBox.IsValid() && !m_bih.bBox.IsInfinite(),
                //    "BIH bounding box is invalid: %s", m_bih.bBox.Format().c_str());
            }

            // If the list contains below the minimum ammount of primitives, don't build and flag the tree as a list traversal
            if (m_hostIndices.size() < m_minBuildablePrims)
            {
                m_bih.testAsList = true;
            }
            else
            {
                // Reserve space for the nodes
                m_hostNodes.reserve(m_hostIndices.size());
                m_hostNodes.resize(1);

                // Construct the bounding interval hierarchy
                m_stats = Stats();
                m_stats.numInnerNodes = 0;

                BuildPartition(0, m_hostIndices.size(), 0, 0, m_bih.bBox, centroidBBox);
            }

            // Update the host data structures
            m_bih.nodes = m_hostNodes.data();
            m_bih.indices = m_hostIndices.data();
            m_bih.numNodes = m_hostNodes.size();
            m_bih.numPrims = m_hostIndices.size();
            m_bih.treeDepth = m_stats.maxDepth;
            m_bih.isConstructed = m_bih.bBox.IsValid() && !m_bih.bBox.IsInfinite();

            if (printStats)
            {
                m_stats.buildTime = timer.Get() * 1e3f;
                Log::Write("Constructed BIH in %.1fms", m_stats.buildTime);
                Log::Write("  - Max depth: %i", m_stats.maxDepth);
                Log::Write("  - Inner nodes: %i", m_stats.numInnerNodes);
                Log::Write("  - Leaf nodes: %i", m_stats.numLeafNodes);
            }
        }

        //template<typename NodeContainer>
        template<typename NodeType>
        __host__ void Builder<NodeType>::BuildPartition(const int i0, const int i1, const uint thisIdx, const uchar depth, const BBox2f& parentBBox, const BBox2f& centroidBBox)
        {
            // Sanity checks
            Assert(depth < 16);

            m_stats.maxDepth = max(depth, m_stats.maxDepth);

            // If the parent bounding box is invalid, warn and discard
            if (!parentBBox.IsValid())
            {
                Log::Warning("Warning: BHV encountered bbox with zero or negative area.");
                m_hostNodes[thisIdx].MakeInvalidLeaf();
                return;
            }

            // If this node only contains one primitive, it's a leaf
            if (i1 - i0 <= 1)
            {
                if (i0 == i1)
                {
                    m_stats.numLeafNodes++;
                    m_hostNodes[thisIdx].MakeInvalidLeaf();
                    //return;
                }
                else if (i1 - i0 == 1)
                {
                    m_stats.numLeafNodes++;
                    m_hostNodes[thisIdx].MakeLeaf(i0, i1);
                    //return;
                }

                /*if (m_debugFunctor)
                {
                    m_debugFunctor("----------\n");
                    m_debugFunctor(tfm::format("Leaf: %i (%i)\n", thisIdx, m_hostNodes[thisIdx].data).c_str());
                    for (int idx = 0; idx < m_hostNodes.Size(); ++idx)
                    {
                        if(m_hostNodes[idx].IsLeaf())
                            m_debugFunctor(tfm::format("  %i -> Leaf %i\n", idx, m_hostNodes[idx].data.primIdxs[0]).c_str());
                        else
                            m_debugFunctor(tfm::format("  %i -> Inner %i, %i\n", idx, m_hostNodes[idx].GetAxis(), m_hostNodes[idx].GetChildIndex()).c_str());
                    }
                    m_debugFunctor("\n");
                }*/
                return;
            }

            const uint axis = centroidBBox.MaxAxis();
            const float split = centroidBBox.Centroid(axis);
            BBox2f leftBBox = BBox2f::Invalid(), rightBBox = BBox2f::Invalid();
            BBox2f leftCentroidBBox = BBox2f::Invalid(), rightCentroidBBox = BBox2f::Invalid();           

            // Sort the primitive indices along the dominant axis
            int j = i0;
            uint sw;
            for (int i = i0; i < i1; ++i)
            {
                const BBox2f primBBox = m_getPrimitiveBBox(m_hostIndices[i]);
                const vec2 primCentroid = primBBox.Centroid();

                if (primCentroid[axis] < split)
                {
                    // Update the left partition position and the moment
                    leftBBox = Union(leftBBox, primBBox);
                    leftCentroidBBox = Union(leftCentroidBBox, primCentroid);
                    // Swap the element into the left-hand partition
                    sw = m_hostIndices[j]; m_hostIndices[j] = m_hostIndices[i]; m_hostIndices[i] = sw;
                    // Increment the partition index
                    ++j;
                }
                else
                {
                    // Update the right partition position and primCentroid
                    rightBBox = Union(rightBBox, primBBox);
                    rightCentroidBBox = Union(rightCentroidBBox, primCentroid);
                }
            }

            // If we've got a bunch of overlapping primitives that we can't effectively partition, just convert this node to a leaf
            if (j == i0 || j == i1)
            {
                m_hostNodes[thisIdx].MakeLeaf(i0, i1);
                return;
            }

            // Grow the node vector by two
            const int leftIdx = m_hostNodes.size();
            const int rightIdx = leftIdx + 1;
            m_hostNodes.Grow(2);

            // Refresh the reference and build the inner node          
            m_hostNodes[thisIdx].MakeInner(leftIdx, axis, leftBBox[1][axis], rightBBox[0][axis], i0, i1);
            m_stats.numInnerNodes++;

            /*if (m_debugFunctor)
            {
                m_debugFunctor("----------\n");
                m_debugFunctor(tfm::format("Inner: %i (%i, %i -> %i)\n", thisIdx, leftIdx, axis, m_hostNodes[thisIdx].data).c_str());
                for (int idx = 0; idx < m_hostNodes.Size(); ++idx)
                {
                    if (m_hostNodes[idx].IsLeaf())
                        m_debugFunctor(tfm::format("  %i -> Leaf %i\n", idx, m_hostNodes[idx].data.primIdxs[0]).c_str());
                    else
                        m_debugFunctor(tfm::format("  %i -> Inner %i, %i\n", idx, m_hostNodes[idx].GetAxis(), m_hostNodes[idx].GetChildIndex()).c_str());
                }
                m_debugFunctor("\n");
            }*/

            // Build the child nodes
            BuildPartition(i0, j, leftIdx, depth + 1, leftBBox, leftCentroidBBox);
            BuildPartition(j, i1, rightIdx, depth + 1, rightBBox, rightCentroidBBox);
        }
    }
}