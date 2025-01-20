#pragma once

#include "Traverser.cuh"

namespace Enso
{
    namespace BIH2D
    {
        // Tests a point with the tree
        // LeafLambda expects the signature bool(const uint*)
        // InnerLambda expects the signature void(const uint*)
        // Returns true if a primitive is successfully hit
        template<typename NodeType, typename LeafLambda, typename InnerLambda = nullptr_t>
        __host__ __device__ static bool TestPoint(const BIHData<NodeType>& bih, const vec2& p, LeafLambda onIntersectLeaf, InnerLambda onIntersectInner = nullptr)
        {
            if (!bih.isConstructed) { return false; }

            // Chuck out primitives that don't overlap the bounding box
            if (!bih.bBox.Intersects(p)) { return false; }

            // If there are only a handful of primitives in the tree, don't bother traversing and just test them all sequentially
            if (bih.testAsList)
            {
                const uint idxRange[2] = { 0, bih.numPrims };
                return onIntersectLeaf(idxRange, bih.indices);
            }

            Traverser::Stack stack;
            BBox2f bBox = bih.bBox;
            NodeType* node = &bih.nodes[0];
            int stackIdx = -1;
            uchar depth = 0;
            bool trace = false;

            CudaAssertDebug(node);

            do
            {
                // If there's no node in place, pop the stack
                if (!node)
                {
                    CudaAssertDebug(stackIdx >= 0);
                    node = &bih.nodes[stack[stackIdx].nodeIdx];
                    depth = stack[stackIdx].depth;
                    bBox = stack[stackIdx--].bBox;
                    CudaAssertDebug(node);
                }

                // Node is a leaf?
                const uchar axis = node->GetAxis();
                if (axis == kBIHLeaf)
                {
                    Traverser::OnPrimitiveIntersectInner(bBox, depth, true, onIntersectInner);
                    if (node->IsValidLeaf())
                    {
                        if (onIntersectLeaf(node->GetPrimIdxs(), bih.indices)) { return true; }
                    }
                    node = nullptr;
                }
                // ...or an inner node.
                else
                {
                    Traverser::OnPrimitiveIntersectInner(bBox, depth, false, onIntersectInner);

                    // Left box hit?
                    if (p[axis] < node->data.planes[NodeType::kLeft])
                        //if(IntersectLeft(p, axis, node->data.planes[NodeType::kLeft], bBox))
                    {
                        // ...and right box hit?
                        if (p[axis] > node->data.planes[NodeType::kRight] && stackIdx < Traverser::kStackSize)
                            //if (IntersectRight(p, axis, node->data.planes[NodeType::kRight], bBox)/* && stackIdx < Traverser::kStackSize*/)
                        {
                            CudaAssertDebug(stackIdx < Traverser::kStackSize);
                            stack[++stackIdx] = { bBox, node->GetChildIndex() + 1, uchar(depth + 1) };
                            stack[stackIdx].bBox[0][axis] = node->data.planes[NodeType::kRight];
                        }

                        bBox[1][axis] = node->data.planes[NodeType::kLeft];
                        node = &bih.nodes[node->GetChildIndex()];
                        ++depth;
                    }
                    // Right box hit?
                    else if (p[axis] > node->data.planes[NodeType::kRight])
                        //else if (IntersectRight(p, axis, node->data.planes[NodeType::kRight], bBox))
                    {
                        bBox[0][axis] = node->data.planes[NodeType::kRight];
                        node = &bih.nodes[node->GetChildIndex() + 1];
                        ++depth;
                    }
                    // Nothing hit. 
                    else
                    {
                        node = nullptr;
                    }
                }
            } while (stackIdx >= 0 || node != nullptr);

            return false;
        }
    }
}