#pragma once

#include "generic/StdIncludes.h"
#include "kernels/math/bbox/CudaBBox2.cuh"
#include <map>

#include "../CudaAsset.cuh"
#include "../CudaManagedObject.cuh"
#include "../CudaImage.cuh"
#include "../CudaVector.cuh"

namespace Cuda
{
    #define kNear 0
    #define kFar 1
    
    enum BIHFlags : unsigned char
    {
        kBIHX = 0,
        kBIHY = 1,
        kBIHLeaf = 2
    };

    struct BIH2DNodeDataFull
    {
        float        planes[2];
        uint         primIdxs[2];
    };

    struct BIH2DNodeDataCompact
    {
        union
        {
            float        planes[2];
            uint         primIdxs[2];
        };
    };

    template<typename NodeDataType>
    struct BIH2DNodeBase
    {
    public:
        enum _attrs : uint { kLeft = 0, kRight = 1, kInvalidLeaf = 0xffffffff };
        
        __host__ __device__ BIH2DNodeBase() = default;

        __host__ __device__ __forceinline__ uchar GetAxis() const { return uchar(flags & uint(3)); }
        __host__ __device__ __forceinline__ uint GetPrimStartIdx() const { return data.primIdxs[0]; }
        __host__ __device__ __forceinline__ uint GetPrimEndIdx() const { return data.primIdxs[1]; }
        __host__ __device__ __forceinline__ uint GetChildIndex() const { return flags >> 2; }
        __host__ __device__ __forceinline__ bool IsValidLeaf() const { return data.primIdxs[0] != kInvalidLeaf; }
        __host__ __device__ __forceinline__ bool IsLeaf() const { return uchar(flags & uint(3)) == kBIHLeaf; }

        __host__ __device__ __forceinline__ BBox2f GetLeftBBox(BBox2f parentBBox) const
        {
            parentBBox.upper[flags & 3u] = data.planes[BIH2DNodeBase::kLeft];
            return parentBBox;
        }

        __host__ __device__ __forceinline__ BBox2f GetRightBBox(BBox2f parentBBox) const
        {
            parentBBox.lower[flags & 3u] = data.planes[BIH2DNodeBase::kRight];
            return parentBBox;
        }
        
        __host__ __device__ __forceinline__ void MakeInner(const uint& i, const uint& split, const float& left, const float& right,
                                                           const uint& primIdxStart, const uint& primIdxEnd)
        { 
            assert(i < ~uint(3));
            flags = (i << 2) | (split & uint(3));
            data.planes[BIH2DNodeBase::kLeft] = left;
            data.planes[BIH2DNodeBase::kRight] = right;

            // If we're using full nodes in this tree, include the start and end indices in the inner node
            if (std::is_same<NodeDataType, BIH2DNodeDataFull>::value)
            {
                data.primIdxs[0] = primIdxStart;
                data.primIdxs[1] = primIdxEnd;
            }
        }

        __host__ __device__ __forceinline__ void MakeLeaf(const uint& idxStart, const uint& idxEnd)
        {
            flags = kBIHLeaf;
            data.primIdxs[0] = idxStart;
            data.primIdxs[1] = idxEnd;
        }

    public:
        uint         flags;
        NodeDataType data;
    };

    using BIH2DCompactNode = BIH2DNodeBase<BIH2DNodeDataCompact>;
    using BIH2DFullNode = BIH2DNodeBase<BIH2DNodeDataFull>;

    template<typename NodeDataType>
    struct BIH2DParams
    {
        bool                        isConstructed = false;
        bool                        testAsList = false;
        BBox2f                      bBox;
        Device::Vector<NodeDataType>*  nodes = nullptr;
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

    using BIH2DNearFar = vec2;

    struct BIH2DPrimitiveStackElement
    {
        BBox2f      bBox;
        uint        nodeIdx;
        uchar       depth;
    };

    struct BIH2DRayStackElement : public BIH2DPrimitiveStackElement
    {
        BIH2DNearFar t;
    };

    namespace Host { class BIH2DBuilder; }

    template<typename TNodeType>
    class BIH2D
    {
        enum _attrs : uint { kMinPrimsInTree = 5 };

    public:

        using NodeType = TNodeType;

        friend class Host::BIH2DBuilder;

        __host__ __device__ BIH2D() :
            m_treeBBox(vec2(0.f), vec2(0.f)),
            m_nodes(nullptr),
            m_isConstructed(false),
            m_testAsList(false),
            m_numNodes(0)
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
                onIntersectLeaf(0, m_numPrims);
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
                        onIntersectLeaf(node->GetPrimStartIdx(), node->GetPrimEndIdx());
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
                onIntersectLeaf(0, m_numPrims, false);
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
                        onIntersectLeaf(node->GetPrimStartIdx(), node->GetPrimEndIdx(), false);
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
                        onIntersectLeaf(node->GetPrimStartIdx(), node->GetPrimEndIdx(), true);
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
        __host__ __device__ __forceinline__ void RayTraverseInnerNode(const vec2& o, const vec2& d, NodeType*& const node, const uchar& axis,
                                                                      BIH2DNearFar& t, BBox2f& bBox, BIH2DRayStackElement* stack, int& stackIdx) const
        {
            constexpr uint kPlaneIdx1 = (kPlaneIdx0 + 1) & 1;
            constexpr float kPlane0Sign = 1.0f - kPlaneIdx0 * 2.0f;
            constexpr float kPlane1Sign = 1.0f - kPlaneIdx1 * 2.0f;

            // Nearest box hit?                
            const float tPlane0 = (node->data.planes[kPlaneIdx0] - o[axis]) / d[axis];
            const float tPlane1 = (node->data.planes[kPlaneIdx1] - o[axis]) / d[axis];
            const uchar hitFlags0 = uchar((o[axis] + d[axis] * t[kNear] - node->data.planes[kPlaneIdx0]) * kPlane0Sign < 0.0f) | (uchar(tPlane0 > t[kNear] && tPlane0 < t[kFar]) << 1);
            const uchar hitFlags1 = uchar((o[axis] + d[axis] * t[kNear] - node->data.planes[kPlaneIdx1]) * kPlane0Sign > 0.0f) | (uchar(tPlane1 > t[kNear] && tPlane1 < t[kFar]) << 1);
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
        __host__ __device__ void TestRay(const vec2& o, const vec2& d, LeafLambda onIntersectLeaf, InnerLambda onIntersectInner = nullptr) const
        {
            if (!m_isConstructed) { return; }

            BIH2DNearFar t;
            if (!TestRayBBox(o, d, m_treeBBox, t)) { return; }

            // If there are only a handful of primitives in the tree, don't bother traversing and just test them all as a list
            float tNearest = t[kFar];
            if (m_testAsList)
            {
                onIntersectLeaf(0, m_numPrims - 1, tNearest);                
                return;
            }

            // Create a stack            
            BIH2DRayStackElement stack[kBIH2DStackSize];
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
                        onIntersectLeaf(node->GetPrimStartIdx(), node->GetPrimEndIdx(), tNearest);
                    }
                    node = nullptr;
                }
                // ...or an inner node.
                else
                {
                    OnRayIntersectInner(bBox, t, false, onIntersectInner);

                    // Left-hand node is likely closer, so traverse that one first
                    if (o[axis] < node->data.planes[NodeType::kLeft])
                    {
                        RayTraverseInnerNode<0>(o, d, node, axis, t, bBox, stack, stackIdx);
                    }
                    // ...otherwise traverse right-hand node first
                    else
                    {
                        RayTraverseInnerNode<1>(o, d, node, axis, t, bBox, stack, stackIdx);
                    } 
                }

                //if (node != nullptr && ++depth > 4) break;
            } 
            while (stackIdx >= 0 || node != nullptr);
        }

        __host__ __device__ __forceinline__ const BBox2f&       GetBBox() const { return m_treeBBox; }
        __host__ __device__ __forceinline__ const NodeType*     GetNodes() const { return m_nodes; }

    protected:
        __host__ __device__ __forceinline__ bool TestRayBBox(const vec2& o, const vec2& d, const BBox2f& bBox, BIH2DNearFar& t) const
        {
            vec2 tNearPlane, tFarPlane;
            for (int dim = 0; dim < 2; dim++)
            {
                if (fabs(d[dim]) > 1e-10f)
                {
                    float t0 = (bBox.upper[dim] - o[dim]) / d[dim];
                    float t1 = (bBox.lower[dim] - o[dim]) / d[dim];
                    if (t0 < t1) { tNearPlane[dim] = t0;  tFarPlane[dim] = t1; }
                    else { tNearPlane[dim] = t1;  tFarPlane[dim] = t0; }
                }
            }

            t[0] = max(0.f, cwiseMax(tNearPlane));
            t[1] = cwiseMin(tFarPlane);
            return t[0] < t[1];
        }

    protected:
        NodeType*                   m_nodes;
        uint                        m_numNodes;
        uint                        m_numPrims;
        BBox2f                      m_treeBBox;

        bool                        m_isConstructed;
        bool                        m_testAsList;
    };   

    using BIH2DFull = BIH2D<BIH2DFullNode>;
    using BIH2DCompact = BIH2D<BIH2DCompactNode>;

    //template BIH2DBuilder<Host::Vector<BIH2DNode>>;

    namespace Device
    {
        class BIH2DAsset : public BIH2DFull, public Device::Asset
        {
        public:
            __device__ BIH2DAsset() {}

            __device__ void Synchronise(const BIH2DParams<BIH2DFullNode>& params);
        };
    }

    namespace Host
    {
        class BIH2DAsset;
        
        //template<typename NodeContainer>
        class BIH2DBuilder
        {
        public:
            __host__ BIH2DBuilder(Host::BIH2DAsset& bih, std::vector<uint>& primitiveIdxs, const uint minBuildablePrims, std::function<BBox2f(uint)>& getPrimitiveBBox) noexcept;

            __host__ void Build();

            std::function<void(const char*)> m_debugFunctor = nullptr;


        protected:
            __host__ void BuildPartition(const int i0, const int i1, const uint thisIdx, const uchar depth, const BBox2f& parentBBox, const BBox2f& centroidBBox);

        private:
            BIH2DFull&                                  m_bih;
            Host::Vector<BIH2DFullNode>&                m_hostNodes;
            std::vector<uint>&                          m_primitiveIdxs;
            std::function<BBox2f(uint)>                 m_getPrimitiveBBox;
            BIH2DStats&                                 m_stats;
            const uint                                  m_minBuildablePrims;
        };
        
        class BIH2DAsset : public BIH2DFull, public Host::Asset
        {
        public:            

            /*template<typename T> */friend class BIH2DBuilder;
            
            __host__ BIH2DAsset(const std::string& id);
            __host__ virtual ~BIH2DAsset();

            __host__ virtual void                   OnDestroyAsset() override final;

            __host__ inline std::vector<uint>&      GetPrimitiveIndices() { return m_primitiveIdxs; }
            __host__ void                           Build(std::function<BBox2f(uint)>& functor);
            __host__ Device::BIH2DAsset*            GetDeviceInstance() const { return cu_deviceInstance; }
            __host__ void                           Synchronise();
            __host__ const BIH2DStats&              GetTreeStats() const { return m_stats; }
            __host__ const Host::Vector<BIH2DFullNode>& GetHostNodes() const { return *m_hostNodes; }

            std::function<void(const char*)> m_debugFunctor = nullptr;

        private:
            __host__ void                           CheckTreeNodes() const;

        private:
            AssetHandle<Host::Vector<BIH2DFullNode>> m_hostNodes;
            std::vector<uint>                       m_primitiveIdxs;
            BIH2DStats                              m_stats;
            BIH2DParams<BIH2DFullNode>              m_params;

            Device::BIH2DAsset*                     cu_deviceInstance;
        };    
    }

    #undef kNear
    #undef kFar
}