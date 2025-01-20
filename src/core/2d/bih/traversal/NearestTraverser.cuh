#pragma once

#include "Traverser.cuh"

namespace Enso
{
    namespace BIH2D
    {
        namespace NearestTraverser
        {
            struct StackElement : public Traverser::StackElement
            {
                float dist;
            };
        }

        template<typename NodeType, typename LeafLambda>
        __host__ __device__ static bool TestNearest(const BIHData<NodeType>& bih, const vec2& p, LeafLambda onIntersectLeaf)
        {
            if (!bih.isConstructed) { return false; }

            // If there are only a handful of primitives in the tree, don't bother traversing and just test them all sequentially
            if (bih.testAsList)
            {
                const uint idxRange[2] = { 0, bih.numPrims };
                return onIntersectLeaf(idxRange, bih.indices);
            }

            NearestTraverser::StackElement stack[Traverser::kStackSize];
            BBox2f bBox = bih.bBox;
            NodeType* node = &bih.nodes[0];
            int stackIdx = -1;
            uchar depth = 0;
            bool trace = false;
            float maxDist = kFltMax;

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
                    if (node->IsValidLeaf())
                    {
                        maxDist = fmin(maxDist, onIntersectLeaf(node->GetPrimIdxs(), bih.indices));
                    }
                    node = nullptr;
                }
                // ...or an inner node.
                else
                {
                    const float leftDist = p[axis] - (node->data.planes[NodeType::kLeft] + maxDist);
                    const float rightDist = (node->data.planes[NodeType::kRight] - maxDist) - p[axis];
                    
                    // Left box hit?
                    if (leftDist < 0)
                    {
                        // ...and right box hit?
                        if (rightDist < 0 && stackIdx < Traverser::kStackSize)
                        {
                            CudaAssertDebug(stackIdx < Traverser::kStackSize);

                            // Traverse whichever child node has the lowest (most negative) distance from the query point to its partition boundary, i.e. whichever one is "deeper" inside.
                            // hitFlag: Left first = 0; right first = 1.
                            const int hitFlag = int(leftDist < rightDist);

                            // Push the "losing" child node onto the stack using the hit flag to index it
                            stack[++stackIdx] = { bBox, node->GetChildIndex() + hitFlag, uchar(depth + 1), (p[axis] - node->data.planes[hitFlag]) * (1 - 2 * hitFlag) };
                            stack[stackIdx].bBox[~hitFlag & 1][axis] = node->data.planes[hitFlag];

                            // Update the current state with the "winning" child node
                            bBox[hitFlag][axis] = node->data.planes[~hitFlag & 1];
                            node = &bih.nodes[node->GetChildIndex() + (~hitFlag & 1)];                           
                            ++depth;
                        }
                        // Just the left box hit
                        else
                        {
                            bBox[1][axis] = node->data.planes[NodeType::kLeft];
                            node = &bih.nodes[node->GetChildIndex()];
                            ++depth;
                        }
                    }
                    // Right box hit?
                    else if (rightDist < 0)
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
            } 
            while (stackIdx >= 0 || node != nullptr);

            return false;
        }
    }
}