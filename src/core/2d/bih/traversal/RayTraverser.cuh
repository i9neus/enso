#pragma once

#include "Traverser.cuh"

namespace Enso
{
    namespace BIH2D
    {
        struct RayTraverser
        {
        private:
            struct RayStackElement : public Traverser::StackElement
            {
                RayRange2D range;
            };

        private:
            template<typename InnerLambda>
            __host__ __device__ __forceinline__ static void OnRayIntersectInner(const BBox2f& bBox, const RayRange2D& t, const bool& isLeaf, InnerLambda onIntersectInner) { onIntersectInner(bBox, t, isLeaf); }
            template<>
            __host__ __device__ __forceinline__ static void OnRayIntersectInner(const BBox2f&, const RayRange2D& t, const bool&, nullptr_t) { }

            template<typename NodeType, uint kPlaneIdx0>
            __host__ __device__ __forceinline__ void RayTraverseInnerNode(const BIHData<NodeType>& bih, const RayBasic2D& ray, NodeType*& node, const uchar& axis,
                                                                          RayRange2D& range, BBox2f& bBox, RayStackElement* stack, int& stackIdx) const
            {
                constexpr uint kPlaneIdx1 = (kPlaneIdx0 + 1) & 1;
                constexpr float kPlane0Sign = 1.0f - kPlaneIdx0 * 2.0f;
                //constexpr float kPlane1Sign = 1.0f - kPlaneIdx1 * 2.0f;

                // Nearest box hit?                
                const float tPlane0 = (node->data.planes[kPlaneIdx0] - ray.o[axis]) / ray.d[axis];
                const float tPlane1 = (node->data.planes[kPlaneIdx1] - ray.o[axis]) / ray.d[axis];
                const uchar hitFlags0 = uchar((ray.o[axis] + ray.d[axis] * range.tNear - node->data.planes[kPlaneIdx0]) * kPlane0Sign < 0.0f) | (uchar(tPlane0 > range.tNear && tPlane0 < range.tFar) << 1);
                const uchar hitFlags1 = uchar((ray.o[axis] + ray.d[axis] * range.tNear - node->data.planes[kPlaneIdx1]) * kPlane0Sign > 0.0f) | (uchar(tPlane1 > range.tNear && tPlane1 < range.tFar) << 1);

                if (hitFlags0)
                {
                    if (hitFlags1/* && stackIdx < kBIH2DStackSize*/)
                    {
                        CudaAssertDebug(stackIdx < kBIH2DStackSize);
                        RayStackElement& element = stack[++stackIdx];
                        element.nodeIdx = node->GetChildIndex() + kPlaneIdx1;
                        element.bBox = bBox;
                        element.bBox[kPlaneIdx0][axis] = node->data.planes[kPlaneIdx1];

                        element.range = range;
                        if (hitFlags1 & 1)
                        {
                            if (tPlane1 > range.tNear) { element.range.tFar = fminf(element.range.tFar, tPlane1); }
                        }
                        else { element.range.tNear = fmaxf(element.range.tNear, tPlane1); }
                    }

                    bBox[kPlaneIdx1][axis] = node->data.planes[kPlaneIdx0];
                    node = &bih.nodes[node->GetChildIndex() + kPlaneIdx0];

                    // Update tNear or tFar depending on whether the ray origin lies inside the bounding box
                    if (hitFlags0 & 1)
                    {
                        if (tPlane0 > range.tNear) { range.tFar = fminf(range.tFar, tPlane0); }
                    }
                    else { range.tNear = fmaxf(range.tNear, tPlane0); }
                }
                // Furthest box hit?
                else if (hitFlags1)
                {
                    bBox[kPlaneIdx0][axis] = node->data.planes[kPlaneIdx1];
                    node = &bih.nodes[node->GetChildIndex() + kPlaneIdx1];

                    if (hitFlags1 & 1)
                    {
                        if (tPlane1 > range.tNear) { range.tFar = fminf(range.tFar, tPlane1); }
                    }
                    else { range.tNear = fmaxf(range.tNear, tPlane1); }
                }
                // Nothing hit. 
                else
                {
                    node = nullptr;
                }
            }

        public:
            template<typename NodeType, typename LeafLambda, typename InnerLambda = nullptr_t>
            __host__ __device__ void Test(const BIHData<NodeType>& bih, const RayBasic2D& ray, const float tFar, LeafLambda onIntersectLeaf, InnerLambda onIntersectInner = nullptr) const
            {
                if (!bih.isConstructed) { return; }

                // If the ray doesn't intersect the bounding box, or the nearest point of intersection is behind tFar, bail out
                RayRange2D range;
                if (!IntersectRayBBox(ray, bih.bBox, range) || range.tNear >= tFar) { return; }

                // Clip the range based on the input value of tFar
                range.ClipFar(tFar);

                // If there are only a handful of primitives in the tree, don't bother traversing and just test them all as a list
                if (bih.testAsList)
                {
                    const uint primsIdxs[2] = { 0, bih.numPrims };
                    onIntersectLeaf(primsIdxs, bih.indices, range);
                    return;
                }

                // Create a stack            
                RayStackElement stack[kStackSize];
                BBox2f bBox = bih.bBox;
                NodeType* node = &bih.nodes[0];
                int stackIdx = -1;
                //uchar depth = 0;

                CudaAssertDebug(node);

                do
                {
                    // If there's no node in place, pop the stack
                    if (!node)
                    {
                        CudaAssertDebug(stackIdx >= 0);
                        const RayStackElement& element = stack[stackIdx--];

                        if (range.tFar < element.range.tNear) { continue; }
                        range = element.range;
                        bBox = element.bBox;
                        node = &bih.nodes[element.nodeIdx];
                        CudaAssertDebug(node);
                    }

                    // Node is a leaf?
                    const uchar axis = node->GetAxis();
                    if (axis == kBIHLeaf)
                    {
                        OnRayIntersectInner(bBox, range, true, onIntersectInner);
                        if (node->IsValidLeaf())
                        {
                            onIntersectLeaf(node->GetPrimIdxs(), bih.indices, range);
                        }
                        node = nullptr;
                    }
                    // ...or an inner node.
                    else
                    {
                        OnRayIntersectInner(bBox, range, false, onIntersectInner);

                        // Left-hand node is likely closer, so traverse that one first
                        if (ray.o[axis] < node->data.planes[NodeType::kLeft])
                        {
                            RayTraverseInnerNode<0>(ray, node, axis, range, bBox, stack, stackIdx);
                        }
                        // ...otherwise traverse right-hand node first
                        else
                        {
                            RayTraverseInnerNode<1>(ray, node, axis, range, bBox, stack, stackIdx);
                        }
                    }

                    //if (node != nullptr && ++depth > 4) break;
                } while (stackIdx >= 0 || node != nullptr);
            }
        };
    }
}