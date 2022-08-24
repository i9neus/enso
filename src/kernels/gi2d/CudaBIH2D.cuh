#pragma once

#include "BIH2DNode.cuh"
#include "GenericIntersector.cuh"

using namespace Cuda;

namespace GI2D
{
    #define kNear 0
    #define kFar 1

    template<typename NodeType> class BIH2DBuilder;

    template<typename NodeDataType>
    struct BIH2DParams
    {
        bool                        isConstructed = false;
        bool                        testAsList = false;
        BBox2f                      bBox;
        Cuda::Device::Vector<NodeDataType>* nodes = nullptr;
        uint                        numPrims = 0;
    };

    struct BIH2DStats
    {
        float buildTime = 0.f;
        uchar maxDepth = 0;
        uint numInnerNodes = 0;
        uint numLeafNodes = 0;
    };

#define kBIH2DStackSize 10

    struct BIH2DPrimitiveStackElement
    {
        BBox2f      bBox;
        uint        nodeIdx;
        uchar       depth;
    };
    
    using BIH2DNearFar = vec2;
    struct BIH2DRayStackElement : public BIH2DPrimitiveStackElement
    {
        BIH2DNearFar t;
    };

    template<typename TNodeType>
    class BIH2D
    {
        enum _attrs : uint { kMinPrimsInTree = 5 };

    public:
        using NodeType = TNodeType;

        template<typename T> friend class BIH2DBuilder;

        __host__ __device__ BIH2D() :
            m_treeBBox(vec2(0.f), vec2(0.f)),
            m_nodes(nullptr),
            m_isConstructed(false),
            m_testAsList(false),
            m_numNodes(0),
            m_numPrims(0)
        {

        }

        __host__ __device__ ~BIH2D() {};

        __host__ __device__ bool IsConstructed() const { return m_isConstructed; }

        template<typename InnerLambda>
        __host__ __device__ __forceinline__ void OnPrimitiveIntersectInner(const BBox2f& bBox, const uchar& depth, InnerLambda onIntersectInner) const { onIntersectInner(bBox, depth); }
        template<>  
        __host__ __device__ __forceinline__ void OnPrimitiveIntersectInner(const BBox2f&, const uchar&, nullptr_t) const { }

        template<typename InnerLambda>
        __host__ __device__ __forceinline__ void OnRayIntersectInner(const BBox2f& bBox, const BIH2DNearFar& t, const bool& isLeaf, InnerLambda onIntersectInner) const { onIntersectInner(bBox, t, isLeaf); }
        template<>
        __host__ __device__ __forceinline__ void OnRayIntersectInner(const BBox2f&, const BIH2DNearFar& t, const bool&, nullptr_t) const { }

        template<typename LeafLambda, typename InnerLambda = nullptr_t>
        __host__ __device__ void TestPoint(const vec2& p, LeafLambda onIntersectLeaf, InnerLambda onIntersectInner = nullptr) const
        {
            if (!m_isConstructed) { return; }

            // Chuck out primitives that don't overlap the bounding box
            if (!m_treeBBox.Intersects(p)) { return; }

            // If there are only a handful of primitives in the tree, don't bother traversing and just test them all sequentially
            if (m_testAsList)
            {
                const uint idxRange[2] = { 0, m_numPrims };
                onIntersectLeaf(idxRange);
                return;
            }

            BIH2DPrimitiveStackElement stack[kBIH2DStackSize];
            BBox2f bBox = m_treeBBox;
            NodeType* node = &m_nodes[0];
            int stackIdx = -1;
            uchar depth = 0;

            assert(node);            

            do
            {                
                // If there's no node in place, pop the stack
                if (!node)
                {
                    assert(stackIdx >= 0); 
                    node = &m_nodes[stack[stackIdx].nodeIdx];
                    depth = stack[stackIdx].depth;
                    bBox = stack[stackIdx--].bBox;                    
                    assert(node);
                }

                // Node is a leaf?
                const uchar axis = node->GetAxis();
                if (axis == kBIHLeaf)
                {
                    OnPrimitiveIntersectInner(bBox, depth, onIntersectInner);
                    if (node->IsValidLeaf())
                    {
                        onIntersectLeaf(node->GetPrimIdxs());
                    }                    
                    node = nullptr;
                }
                // ...or an inner node.
                else
                {
                    OnPrimitiveIntersectInner(bBox, depth, onIntersectInner);
                    
                    // Left box hit?
                    if(p[axis] < node->data.planes[NodeType::kLeft])
                        //if(IntersectLeft(p, axis, node->data.planes[NodeType::kLeft], bBox))
                    {
                        // ...and right box hit?
                        if (p[axis] > node->data.planes[NodeType::kRight] && stackIdx < kBIH2DStackSize)
                            //if (IntersectRight(p, axis, node->data.planes[NodeType::kRight], bBox)/* && stackIdx < kBIH2DStackSize*/)
                        {
                            assert(stackIdx < kBIH2DStackSize);
                            stack[++stackIdx] = { bBox, node->GetChildIndex() + 1, uchar(depth + 1) };
                            stack[stackIdx].bBox[0][axis] = node->data.planes[NodeType::kRight];
                        }

                        bBox[1][axis] = node->data.planes[NodeType::kLeft];
                        node = &m_nodes[node->GetChildIndex()];
                        ++depth;
                    }
                    // Right box hit?
                    else if (p[axis] > node->data.planes[NodeType::kRight])
                        //else if (IntersectRight(p, axis, node->data.planes[NodeType::kRight], bBox))
                    {
                        bBox[0][axis] = node->data.planes[NodeType::kRight];
                        node = &m_nodes[node->GetChildIndex() + 1];
                        ++depth;
                    }
                    // Nothing hit. 
                    else
                    {
                        node = nullptr;
                    }
                }
            } while (stackIdx >= 0 || node != nullptr);
        }

        template<typename LeafLambda, typename InnerLambda = nullptr_t>
        __host__ __device__ void TestBBox(const BBox2f& p, LeafLambda onIntersectLeaf, InnerLambda onIntersectInner = nullptr) const
        {
            if (!m_isConstructed) { return; }

            // Chuck out primitives that don't overlap the bounding box
            if (!m_treeBBox.Intersects(p)) { return; }

            // If there are only a handful of primitives in the tree, don't bother traversing and just test them all sequentially
            if (m_testAsList)
            {
                const uint idxRange[2] = { 0, m_numPrims };
                onIntersectLeaf(idxRange, false);
                return;
            }

            BIH2DPrimitiveStackElement stack[kBIH2DStackSize];
            BBox2f nodeBBox = m_treeBBox;
            NodeType* node = &m_nodes[0];
            int stackIdx = -1;
            uchar depth = 0;

            assert(node);

            do
            {
                // If there's no node in place, pop the stack
                if (!node)
                {
                    assert(stackIdx >= 0);
                    node = &m_nodes[stack[stackIdx].nodeIdx];
                    depth = stack[stackIdx].depth;
                    nodeBBox = stack[stackIdx--].bBox;
                    assert(node);
                }

                // Node is a leaf?
                const uchar axis = node->GetAxis();
                if (axis == kBIHLeaf)
                {
                    OnPrimitiveIntersectInner(nodeBBox, depth, onIntersectInner);
                    if (node->IsValidLeaf())
                    {
                        onIntersectLeaf(node->GetPrimIdxs(), false);
                    }
                    node = nullptr;
                }
                // ...or an inner node.
                else
                {
                    OnPrimitiveIntersectInner(nodeBBox, depth, onIntersectInner);

                    // If the entire node is contained within p, don't traverse the tree any further. Instead, just invoke the functor for all primitives contained by the node
                    if (p.Contains(nodeBBox))
                    {
                        onIntersectLeaf(node->GetPrimIdxs(), true);
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
                                assert(stackIdx < kBIH2DStackSize);
                                stack[++stackIdx] = { nodeBBox, node->GetChildIndex() + 1, uchar(depth + 1) };
                                stack[stackIdx].bBox[0][axis] = node->data.planes[NodeType::kRight];
                            }

                            nodeBBox[1][axis] = node->data.planes[NodeType::kLeft];
                            node = &m_nodes[node->GetChildIndex()];
                            ++depth;
                        }
                        // Right box hit?
                        else if (p.upper[axis] > node->data.planes[NodeType::kRight])
                        {
                            nodeBBox[0][axis] = node->data.planes[NodeType::kRight];
                            node = &m_nodes[node->GetChildIndex() + 1];
                            ++depth;
                        }
                        // Nothing hit.
                        else
                        {
                            node = nullptr;
                        }
                    }
                }
            } 
            while (stackIdx >= 0 || node != nullptr);
        }

        template<uint kPlaneIdx0>
        __host__ __device__ __forceinline__ void RayTraverseInnerNode(const RayBasic2D& ray, NodeType*& node, const uchar& axis,
                                                                      BIH2DNearFar& t, BBox2f& bBox, BIH2DRayStackElement* stack, int& stackIdx) const
        {
            constexpr uint kPlaneIdx1 = (kPlaneIdx0 + 1) & 1;
            constexpr float kPlane0Sign = 1.0f - kPlaneIdx0 * 2.0f;
            //constexpr float kPlane1Sign = 1.0f - kPlaneIdx1 * 2.0f;

            // Nearest box hit?                
            const float tPlane0 = (node->data.planes[kPlaneIdx0] - ray.o[axis]) / ray.d[axis];
            const float tPlane1 = (node->data.planes[kPlaneIdx1] - ray.o[axis]) / ray.d[axis];
            const uchar hitFlags0 = uchar((ray.o[axis] + ray.d[axis] * t[kNear] - node->data.planes[kPlaneIdx0]) * kPlane0Sign < 0.0f) | (uchar(tPlane0 > t[kNear] && tPlane0 < t[kFar]) << 1);
            const uchar hitFlags1 = uchar((ray.o[axis] + ray.d[axis] * t[kNear] - node->data.planes[kPlaneIdx1]) * kPlane0Sign > 0.0f) | (uchar(tPlane1 > t[kNear] && tPlane1 < t[kFar]) << 1);
            if (hitFlags0)
            {
                if (hitFlags1/* && stackIdx < kBIH2DStackSize*/)
                {
                    assert(stackIdx < kBIH2DStackSize);
                    BIH2DRayStackElement& element = stack[++stackIdx];
                    element.nodeIdx = node->GetChildIndex() + kPlaneIdx1;
                    element.bBox = bBox;
                    element.bBox[kPlaneIdx0][axis] = node->data.planes[kPlaneIdx1];

                    element.t = t;
                    if (hitFlags1 & 1)
                    {
                        if (tPlane1 > t[kNear]) { element.t[kFar] = min(element.t[kFar], tPlane1); }
                    }
                    else { element.t[kNear] = max(element.t[kNear], tPlane1); }
                }

                bBox[kPlaneIdx1][axis] = node->data.planes[kPlaneIdx0];
                node = &m_nodes[node->GetChildIndex() + kPlaneIdx0];

                // Update tNear or tFar depending on whether the ray origin lies inside the bounding box
                if (hitFlags0 & 1)
                {
                    if (tPlane0 > t[kNear]) { t[kFar] = min(t[kFar], tPlane0); }
                }
                else { t[kNear] = max(t[kNear], tPlane0); }
            }
            // Furthest box hit?
            else if (hitFlags1)
            {
                bBox[kPlaneIdx0][axis] = node->data.planes[kPlaneIdx1];
                node = &m_nodes[node->GetChildIndex() + kPlaneIdx1];

                if (hitFlags1 & 1)
                {
                    if (tPlane1 > t[kNear]) { t[kFar] = min(t[kFar], tPlane1); }
                }
                else { t[kNear] = max(t[kNear], tPlane1); }
            }
            // Nothing hit. 
            else
            {
                node = nullptr;
            }
        }

        template<typename LeafLambda, typename InnerLambda = nullptr_t>
        __host__ __device__ void TestRay(const RayBasic2D& ray, LeafLambda onIntersectLeaf, InnerLambda onIntersectInner = nullptr) const
        {
            if (!m_isConstructed) { return; }

            BIH2DNearFar t;
            if (!IntersectRayBBox(ray, m_treeBBox, t)) { return; }

            // If there are only a handful of primitives in the tree, don't bother traversing and just test them all as a list
            float tNearest = t[kFar];
            if (m_testAsList)
            {
                const uint primsIdxs[2] = { 0, m_numPrims };
                onIntersectLeaf(primsIdxs, tNearest);
                return;
            }

            // Create a stack            
            BIH2DRayStackElement stack[kBIH2DStackSize];
            BBox2f bBox = m_treeBBox;
            NodeType* node = &m_nodes[0];
            int stackIdx = -1;
            //uchar depth = 0;

            assert(node);

            do
            {
                // If there's no node in place, pop the stack
                if (!node)
                {
                    assert(stackIdx >= 0);
                    const BIH2DRayStackElement& element = stack[stackIdx--];

                    if (tNearest < element.t[kNear]) { continue; }
                    t = element.t;
                    bBox = element.bBox;
                    node = &m_nodes[element.nodeIdx];
                    assert(node);
                }

                // Node is a leaf?
                const uchar axis = node->GetAxis();
                if (axis == kBIHLeaf)
                {
                    OnRayIntersectInner(bBox, t, true, onIntersectInner);
                    if (node->IsValidLeaf())
                    {
                        onIntersectLeaf(node->primsIdxs, tNearest);
                    }
                    node = nullptr;
                }
                // ...or an inner node.
                else
                {
                    OnRayIntersectInner(bBox, t, false, onIntersectInner);

                    // Left-hand node is likely closer, so traverse that one first
                    if (ray.o[axis] < node->data.planes[NodeType::kLeft])
                    {
                        RayTraverseInnerNode<0>(ray, node, axis, t, bBox, stack, stackIdx);
                    }
                    // ...otherwise traverse right-hand node first
                    else
                    {
                        RayTraverseInnerNode<1>(ray, node, axis, t, bBox, stack, stackIdx);
                    } 
                }

                //if (node != nullptr && ++depth > 4) break;
            } 
            while (stackIdx >= 0 || node != nullptr);
        }

        __host__ __device__ __forceinline__ const BBox2f&       GetBoundingBox() const { return m_treeBBox; }
        __host__ __device__ __forceinline__ const NodeType*     GetNodes() const { return m_nodes; }
        __host__ __device__ __forceinline__ uint                GetNumPrimitives() const { return m_numPrims; }

    protected:
        NodeType*                   m_nodes;
        uint                        m_numNodes;
        uint                        m_numPrims;
        BBox2f                      m_treeBBox;

        bool                        m_isConstructed;
        bool                        m_testAsList;

        BIH2DStats                  m_stats;
    };   

    //using BIH2DFull = BIH2D<BIH2DFullNode>;
    //using BIH2DCompact = BIH2D<BIH2DCompactNode>;

    //template BIH2DBuilder<Host::Vector<BIH2DNode>>;

    #undef kNear
    #undef kFar
}