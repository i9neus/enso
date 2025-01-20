#pragma once

#include "Traverser.cuh"

namespace Enso
{
    namespace BIH2D
    {
        // Tests a bounding box with the tree
        template<typename NodeType, typename LeafLambda, typename InnerLambda = nullptr_t>
        __host__ __device__ static void TestBox(const BIHData<NodeType>& bih, const BBox2f& p, LeafLambda onIntersectLeaf, InnerLambda onIntersectInner = nullptr)
        {
            if (!bih.isConstructed) { return; }

            // Chuck out primitives that don't overlap the bounding box
            if (!bih.bBox.Intersects(p)) { return; }

            // If there are only a handful of primitives in the tree, don't bother traversing and just test them all sequentially
            if (bih.testAsList)
            {
                const uint idxRange[2] = { 0, bih.numPrims };
                onIntersectLeaf(idxRange, bih.indices, false);
                return;
            }

            Traverser::Stack stack;
            BBox2f nodeBBox = bih.bBox;
            NodeType* node = &bih.nodes[0];
            int stackIdx = -1;
            uchar depth = 0;

            CudaAssertDebug(node);

            do
            {
                // If there's no node in place, pop the stack
                if (!node)
                {
                    CudaAssertDebug(stackIdx >= 0);
                    node = &bih.nodes[stack[stackIdx].nodeIdx];
                    depth = stack[stackIdx].depth;
                    nodeBBox = stack[stackIdx--].bBox;
                    CudaAssertDebug(node);
                }

                // Node is a leaf?
                const uchar axis = node->GetAxis();
                if (axis == kBIHLeaf)
                {
                    Traverser::OnPrimitiveIntersectInner(nodeBBox, depth, true, onIntersectInner);
                    if (node->IsValidLeaf())
                    {
                        onIntersectLeaf(node->GetPrimIdxs(), bih.indices, false);
                    }
                    node = nullptr;
                }
                // ...or an inner node.
                else
                {
                    Traverser::OnPrimitiveIntersectInner(nodeBBox, depth, false, onIntersectInner);

                    // If the entire node is contained within p, don't traverse the tree any further. Instead, just invoke the functor for all primitives contained by the node
                    if (p.Contains(nodeBBox))
                    {
                        onIntersectLeaf(node->GetPrimIdxs(), bih.indices, true);
                        node = nullptr;
                    }
                    else
                    {
                        // Left box hit?
                        if (p.lower[axis] < node->data.planes[NodeType::kLeft])
                        {
                            // ...and right box hit?
                            if (p.upper[axis] > node->data.planes[NodeType::kRight])
                            {
                                CudaAssertDebug(stackIdx < Traverser::kStackSize);
                                stack[++stackIdx] = { nodeBBox, node->GetChildIndex() + 1, uchar(depth + 1) };
                                stack[stackIdx].bBox[0][axis] = node->data.planes[NodeType::kRight];
                            }

                            nodeBBox[1][axis] = node->data.planes[NodeType::kLeft];
                            node = &bih.nodes[node->GetChildIndex()];
                            ++depth;
                        }
                        // Right box hit?
                        else if (p.upper[axis] > node->data.planes[NodeType::kRight])
                        {
                            nodeBBox[0][axis] = node->data.planes[NodeType::kRight];
                            node = &bih.nodes[node->GetChildIndex() + 1];
                            ++depth;
                        }
                        // Nothing hit.
                        else
                        {
                            node = nullptr;
                        }
                    }
                }
            } while (stackIdx >= 0 || node != nullptr);
        }

    }
}