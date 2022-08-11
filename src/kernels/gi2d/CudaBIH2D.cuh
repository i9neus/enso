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

    struct BIH2DNode
    {
    public:
        enum _attrs : uint { kLeft = 0, kRight = 1, kInvalidLeaf = 0xffffffff };
        
        __host__ __device__ BIH2DNode() = default;

        __host__ __device__ __forceinline__ uchar GetAxis() const { return uchar(data & 3u); }
        __host__ __device__ __forceinline__ uint GetPrimIndex() const { return idx; }
        __host__ __device__ __forceinline__ uint GetChildIndex() const { return data >> 2; }
        __host__ __device__ __forceinline__ bool IsValidLeaf() const { return idx != kInvalidLeaf; }
        __host__ __device__ __forceinline__ bool IsLeaf() const { return uchar(data & 3u) == kBIHLeaf; }

        __host__ __device__ __forceinline__ BBox2f GetLeftBBox(BBox2f parentBBox) const
        {
            parentBBox.upper[data & 3u] = planes[BIH2DNode::kLeft];
            return parentBBox;
        }

        __host__ __device__ __forceinline__ BBox2f GetRightBBox(BBox2f parentBBox) const
        {
            parentBBox.lower[data & 3u] = planes[BIH2DNode::kRight];
            return parentBBox;
        }
        
        __host__ __device__ __forceinline__ void MakeInner(const uint& i, const uint& split, const float& left, const float& right) 
        { 
            assert(i < ~3u);
            data = (i << 2) | (split & 3u);
            planes[BIH2DNode::kLeft] = left;
            planes[BIH2DNode::kRight] = right;
        }

        __host__ __device__ __forceinline__ void MakeLeaf(const uint& i)
        {
            data = kBIHLeaf;
            idx = i;
        }

    public:
        uint            data;
        union
        {
            float           planes[2];            
            uint            idx;
        };
    };

    struct BIH2DParams
    {
        bool                        isConstructed;
        BBox2f                      bBox;
        Device::Vector<BIH2DNode>*  nodes;
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

    class BIH2D
    {
    public:
        friend class Host::BIH2DBuilder;

        __host__ __device__ BIH2D();
        __host__ __device__ ~BIH2D() {};

        __host__ __device__ bool IsConstructed() const { return m_isConstructed; }

        // Intersect point
        __host__ __device__ __forceinline__ bool IntersectLeft(const vec2& p, const uchar& axis, const float& left, const BBox2f&) const { return p[axis] < left; }
        __host__ __device__ __forceinline__ bool IntersectRight(const vec2& p, const uchar& axis, const float& right, const BBox2f&) const { return p[axis] > right; }

        // Intersect bounding box
        __host__ __device__ __forceinline__ bool IntersectLeft(const BBox2f& p, const uchar& axis, const float& left, const BBox2f& bBox) const
        { 
            return max(p.lower[axis], bBox.lower[axis]) - min(p.upper[axis], left) > 0.0f;           
        }
        __host__ __device__ __forceinline__ bool IntersectRight(const BBox2f& p, const uchar& axis, const float& right, const BBox2f& bBox) const
        { 
            return max(p.lower[axis], right) - min(p.upper[axis], bBox.upper[axis]) > 0.0f;
        }

        template<typename InnerLambda>
        __host__ __device__ __forceinline__ void OnPrimitiveIntersectInner(const BBox2f& bBox, const uchar& depth, InnerLambda onIntersectInner) const { onIntersectInner(bBox, depth); }
        template<>  
        __host__ __device__ __forceinline__ void OnPrimitiveIntersectInner(const BBox2f&, const uchar&, nullptr_t) const { }

        template<typename InnerLambda>
        __host__ __device__ __forceinline__ void OnRayIntersectInner(const BBox2f& bBox, const BIH2DNearFar& t, const bool& isLeaf, InnerLambda onIntersectInner) const { onIntersectInner(bBox, t, isLeaf); }
        template<>
        __host__ __device__ __forceinline__ void OnRayIntersectInner(const BBox2f&, const BIH2DNearFar& t, const bool&, nullptr_t) const { }

        template<typename TestType, typename LeafLambda, typename InnerLambda = nullptr_t>
        __host__ __device__ void TestPrimitive(const TestType& p, LeafLambda onIntersectLeaf, InnerLambda onIntersectInner = nullptr) const
        {
            if (!m_isConstructed) { return; }

            BIH2DPrimitiveStackElement stack[kBIH2DStackSize];
            BBox2f bBox = m_bBox;
            BIH2DNode* node = &m_nodes[0];
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
                        onIntersectLeaf(node->GetPrimIndex());
                    }                    
                    node = nullptr;
                }
                // ...or an inner node.
                else
                {
                    OnPrimitiveIntersectInner(bBox, depth, onIntersectInner);
                    
                    // Left box hit?
                    //if(p[axis] < node->left)
                    if(IntersectLeft(p, axis, node->planes[BIH2DNode::kLeft], bBox))
                    {
                        // ...and right box hit?
                        //if (p[axis] > node->right && stackIdx < kBIH2DStackSize)
                        if (IntersectRight(p, axis, node->planes[BIH2DNode::kRight], bBox) && stackIdx < kBIH2DStackSize)
                        {
                            stack[++stackIdx] = { bBox, node->GetChildIndex() + 1, uchar(depth + 1) };
                            stack[stackIdx].bBox[0][axis] = node->planes[BIH2DNode::kRight];
                        }
                        
                        bBox[1][axis] = node->planes[BIH2DNode::kLeft];
                        node = &m_nodes[node->GetChildIndex()];
                        ++depth;
                    }
                    // Right box hit?
                    else if (IntersectRight(p, axis, node->planes[BIH2DNode::kRight], bBox))
                    {
                        bBox[0][axis] = node->planes[BIH2DNode::kRight];
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
            while (stackIdx >= 0 || node != nullptr);
        }    
       
        template<uint kFirstPlaneIdx>
        __host__ __device__ __forceinline__ void RayTraverseInnerNode(const vec2& o, const vec2& d, BIH2DNode*& const node, const uchar& axis, 
                                                                      BIH2DNearFar& t, BBox2f& bBox, BIH2DRayStackElement* stack, int& stackIdx) const
        {            
            constexpr uint kLastPlaneIdx = (kFirstPlaneIdx + 1) & 1;
            constexpr float kFirstPlaneIdxSign = 1.0f - kFirstPlaneIdx * 2.0f;
            constexpr float kLastPlaneIdxSign = 1.0f - kLastPlaneIdx * 2.0f;
            
            // Nearest box hit?                
            const float tPlaneFirst = (node->planes[kFirstPlaneIdx] - o[axis]) / d[axis];
            const bool isInFirst = (node->planes[kFirstPlaneIdx] - o[axis]) * kFirstPlaneIdxSign > 0.f;
            if (isInFirst || (tPlaneFirst > t[0] && tPlaneFirst < t[1]))
            {
                // ...and furthest box hit?
                const float tPlaneLast = (node->planes[kLastPlaneIdx] - o[axis]) / d[axis];
                const bool isInLast = (node->planes[kLastPlaneIdx] - o[axis]) * kLastPlaneIdxSign > 0.f;
                if (isInLast || (tPlaneLast > t[0] && tPlaneLast < t[1]))
                {
                    BIH2DRayStackElement& element = stack[++stackIdx];
                    element.nodeIdx = node->GetChildIndex() + kLastPlaneIdx;
                    element.bBox = bBox;
                    element.bBox[kFirstPlaneIdx][axis] = node->planes[kLastPlaneIdx];
                    element.t = t;
                   
                    if (isInLast)
                    {
                        if (tPlaneLast > 0.f) element.t[1] = min(element.t[1], tPlaneLast);
                    }
                    else
                    {
                        element.t[0] = tPlaneLast;
                    }
                }

                bBox[kLastPlaneIdx][axis] = node->planes[kFirstPlaneIdx];
                node = &m_nodes[node->GetChildIndex() + kFirstPlaneIdx];

                // Update tNear or tFar depending on whether the ray origin lies inside the bounding box
                if (isInFirst)
                {
                    if (tPlaneFirst > 0.f) t[1] = min(t[1], tPlaneFirst);
                }
                else
                {
                    t[0] = tPlaneFirst;
                }
            }
            // Furthest box hit?
            else
            {
                const float tPlaneLast = (node->planes[kLastPlaneIdx] - o[axis]) / d[axis];
                const bool isInLast = (node->planes[kLastPlaneIdx] - o[axis]) * kLastPlaneIdxSign > 0.f;
                if (isInLast || (tPlaneLast > t[0] && tPlaneLast < t[1]))
                {
                    bBox[kFirstPlaneIdx][axis] = node->planes[kLastPlaneIdx];
                    node = &m_nodes[node->GetChildIndex() + kLastPlaneIdx];

                    // Update tNear or tFar depending on whether the ray origin lies inside the bounding box
                    if (isInLast)
                    {
                        if (tPlaneLast > 0.f) t[1] = min(t[1], tPlaneLast);
                    }
                    else
                    {
                        t[0] = tPlaneLast;
                    }
                }
                // Nothing hit. 
                else
                {
                    node = nullptr;
                }
            }

            /*if (kKernelIdx == 0)
            {
                printf("%f -> {{%f, %f}, {%f, %f}}\n", node->planes[kFirstPlaneIdx], bBox[0][0], bBox[0][1], bBox[1][0], bBox[1][1]);
            }*/
        }

        template<uint kPlaneIdx0>
        __host__ __device__ __forceinline__ void RayTraverseInnerNodeLegacy(const vec2& o, const vec2& d, BIH2DNode*& const node, const uchar& axis, 
                                                                            BIH2DNearFar& t, BBox2f& bBox, BIH2DRayStackElement* stack, int& stackIdx) const
        {
            constexpr uint kPlaneIdx1 = (kPlaneIdx0 + 1) & 1;
            constexpr float kPlane0Sign = 1.0f - kPlaneIdx0 * 2.0f;
            constexpr float kPlane1Sign = 1.0f - kPlaneIdx1 * 2.0f;
            
            // Nearest box hit?                
            const float tPlane0 = (node->planes[kPlaneIdx0] - o[axis]) / d[axis];
            const float tPlane1 = (node->planes[kPlaneIdx1] - o[axis]) / d[axis];
            const uchar hitFlags0 = uchar((o[axis] + d[axis] * t[kNear] - node->planes[kPlaneIdx0]) * kPlane0Sign < 0.0f) | (uchar(tPlane0 > t[kNear] && tPlane0 < t[kFar]) << 1);
            const uchar hitFlags1 = uchar((o[axis] + d[axis] * t[kNear] - node->planes[kPlaneIdx1]) * kPlane0Sign > 0.0f) | (uchar(tPlane1 > t[kNear] && tPlane1 < t[kFar]) << 1);
            if (hitFlags0)
            {
                if (hitFlags1)
                {
                    BIH2DRayStackElement& element = stack[++stackIdx];
                    element.nodeIdx = node->GetChildIndex() + kPlaneIdx1;
                    element.bBox = bBox;
                    element.bBox[kPlaneIdx0][axis] = node->planes[kPlaneIdx1];
                    
                    element.t = t;
                    if (hitFlags1 & 1)
                    {
                        if (tPlane1 > t[kNear]) { element.t[kFar] = min(element.t[kFar], tPlane1); }
                    }
                    else                        { element.t[kNear] = max(element.t[kNear], tPlane1); }
                }   

                bBox[kPlaneIdx1][axis] = node->planes[kPlaneIdx0];
                node = &m_nodes[node->GetChildIndex() + kPlaneIdx0];
                
                // Update tNear or tFar depending on whether the ray origin lies inside the bounding box
                if (hitFlags0 & 1)
                {
                    if (tPlane0 > t[kNear]) { t[kFar] = min(t[kFar], tPlane0); }
                }
                else                        { t[kNear] = max(t[kNear], tPlane0); }            
            }
            // Furthest box hit?
            else if (hitFlags1)
            {
                bBox[kPlaneIdx0][axis] = node->planes[kPlaneIdx1];
                node = &m_nodes[node->GetChildIndex() + kPlaneIdx1];

                if (hitFlags1 & 1)
                {
                    if (tPlane1 > t[kNear]) { t[kFar] = min(t[kFar], tPlane1); }
                }                
                else                        { t[kNear] = max(t[kNear], tPlane1); }
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
            if(!TestRayBBox(o, d, m_bBox, t)) { return; }

            // Create a stack            
            BIH2DRayStackElement stack[kBIH2DStackSize];
            BBox2f bBox = m_bBox;
            BIH2DNode* node = &m_nodes[0];
            int stackIdx = -1;
            uchar depth = 0;
            float tNearest = kFltMax;

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
                        onIntersectLeaf(node->GetPrimIndex(), tNearest);
                    }
                    node = nullptr;
                }
                // ...or an inner node.
                else
                {   
                    OnRayIntersectInner(bBox, t, false, onIntersectInner);
                    
                    // Left-hand node is likely closer, so traverse that one first
                    if (o[axis] < node->planes[BIH2DNode::kLeft])
                    {
                        RayTraverseInnerNodeLegacy<0>(o, d, node, axis, t, bBox, stack, stackIdx);
                    }
                    // ...otherwise traverse right-hand node first
                    else
                    {
                        RayTraverseInnerNodeLegacy<1>(o, d, node, axis, t, bBox, stack, stackIdx);
                    } 
                    //RayTraverseInnerNodeLegacy<0>(o, d, node, axis, t, bBox, stack, stackIdx);
                }

                //if (node != nullptr && ++depth > 4) break;
            } 
            while (stackIdx >= 0 || node != nullptr);
        }

        __host__ __device__ __forceinline__ const BBox2f&       GetBBox() const { return m_bBox; }
        __host__ __device__ __forceinline__ const BIH2DNode*    GetNodes() const { return m_nodes; }

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
        BIH2DNode*                  m_nodes;
        uint                        m_numNodes;
        BBox2f                      m_bBox;
        bool                        m_isConstructed;
    };   

    //template BIH2DBuilder<Host::Vector<BIH2DNode>>;

    namespace Device
    {
        class BIH2DAsset : public BIH2D, public Device::Asset
        {
        public:
            __device__ BIH2DAsset() {}

            __device__ void Synchronise(const BIH2DParams& params);
        };
    }

    namespace Host
    {
        class BIH2DAsset;
        
        //template<typename NodeContainer>
        class BIH2DBuilder
        {
        public:
            __host__ BIH2DBuilder(Host::BIH2DAsset& bih, std::vector<uint>& primitiveIdxs, std::function<BBox2f(uint)>& functor) noexcept;

            __host__ void Build();

            std::function<void(const char*)> m_debugFunctor = nullptr;


        protected:
            __host__ void BuildPartition(const int i0, const int i1, const uint thisIdx, const uchar depth, const BBox2f& parentBBox, const BBox2f& centroidBBox);

        private:
            BIH2D&                                      m_bih;
            Host::Vector<BIH2DNode>&                    m_hostNodes;
            std::vector<uint>&                          m_primitiveIdxs;
            std::function<BBox2f(uint)>                 m_getPrimitiveBBox;
            BIH2DStats&                                 m_stats;
        };
        
        class BIH2DAsset : public BIH2D, public Host::Asset
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
            __host__ const Host::Vector<BIH2DNode>& GetHostNodes() const { return *m_hostNodes; }

            std::function<void(const char*)> m_debugFunctor = nullptr;

        private:
            __host__ void                           CheckTreeNodes() const;

        private:
            AssetHandle<Host::Vector<BIH2DNode>>    m_hostNodes;
            std::vector<uint>                       m_primitiveIdxs;
            BIH2DStats                              m_stats;

            Device::BIH2DAsset*                     cu_deviceInstance;
        };    
    }

    #undef kNear
    #undef kFar
}