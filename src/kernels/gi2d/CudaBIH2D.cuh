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

        template<typename TestType, typename LeafLambda/*, typename InnerLambda*/>
        __host__ __device__ void TestPrimitive(const TestType& p, LeafLambda onIntersectLeaf/*, InnerLambda onIntersectInner*/) const
        {
            if (!m_isConstructed) { return; }

            BIH2DPrimitiveStackElement stack[kBIH2DStackSize];
            BBox2f bBox = m_bBox;
            BIH2DNode* node = &m_nodes[0];
            int stackIdx = -1;

            assert(node);            

            do
            {                
                // If there's no node in place, pop the stack
                if (!node)
                {
                    assert(stackIdx >= 0); 
                    node = &m_nodes[stack[stackIdx].nodeIdx];
                    bBox = stack[stackIdx--].bBox;                    
                    assert(node);
                }

                // Node is a leaf?
                const uchar axis = node->GetAxis();
                if (axis == kBIHLeaf)
                {
                    if (node->IsValidLeaf())
                    {
                        onIntersectLeaf(node->GetPrimIndex());
                    }
                    node = nullptr;
                }
                // ...or an inner node.
                else
                {
                    // Left box hit?
                    //if(p[axis] < node->left)
                    if(IntersectLeft(p, axis, node->planes[BIH2DNode::kLeft], bBox))
                    {
                        // ...and right box hit?
                        //if (p[axis] > node->right && stackIdx < kBIH2DStackSize)
                        if (IntersectRight(p, axis, node->planes[BIH2DNode::kRight], bBox) && stackIdx < kBIH2DStackSize)
                        {
                            stack[++stackIdx] = { bBox, node->GetChildIndex() + 1 };
                            stack[stackIdx].bBox[0][axis] = node->planes[BIH2DNode::kRight];
                        }

                        node = &m_nodes[node->GetChildIndex()];
                        bBox[1][axis] = node->planes[BIH2DNode::kLeft];
                    }
                    // Right box hit?
                    else if (IntersectRight(p, axis, node->planes[BIH2DNode::kRight], bBox))
                    {
                        node = &m_nodes[node->GetChildIndex() + 1];
                        bBox[0][axis] = node->planes[BIH2DNode::kRight];
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
       
        __host__ __device__ __forceinline__ void RayTraverseInnerNode(const vec2& o, const vec2& d, BIH2DNode*& const node, const uchar& axis, const uchar& nearPlaneIdx,
                                                                      BIH2DNearFar& t, BBox2f& bBox, BIH2DRayStackElement* stack, int& stackIdx) const
        {
            // Nearest box hit?    
            const float tPlaneNear = (node->planes[nearPlaneIdx] - o[axis]) / d[axis];
            if (tPlaneNear > t[0] && tPlaneNear < t[1])
            {
                // ...and furthest box hit?
                const float tPlaneFar = (node->planes[(nearPlaneIdx + 1) & 1] - o[axis]) / d[axis];
                if (tPlaneFar > t[0] && tPlaneFar < t[1])
                {
                    BIH2DRayStackElement& element = stack[++stackIdx];
                    element.nodeIdx = node->GetChildIndex() + 1;
                    element.bBox = bBox;
                    element.bBox[0][axis] = node->planes[(nearPlaneIdx + 1) & 1];

                    // Update tNear or tFar depending on whether the ray origin lies inside the bounding box
                    if (element.bBox.Contains(o)) { element.t[1] = tPlaneNear; element.t[0] = 0.0f; }
                    else { element.t[0] = tPlaneNear; }
                }

                node = &m_nodes[node->GetChildIndex()];
                bBox[1][axis] = node->planes[nearPlaneIdx];
                
                // Update tNear or tFar depending on whether the ray origin lies inside the bounding box
                if (bBox.Contains(o)) { t[1] = tPlaneNear; t[0] = 0.0f; }
                else { t[0] = tPlaneNear; }
            }
            // Furthest box hit?
            else
            {
                const float tPlaneFar = (node->planes[(nearPlaneIdx + 1) & 1] - o[axis]) / d[axis];
                if (tPlaneFar > t[0] && tPlaneFar < t[1])
                {
                    node = &m_nodes[node->GetChildIndex() + 1];
                    bBox[0][axis] = node->planes[(nearPlaneIdx + 1) & 1];
                    
                    // Update tNear or tFar depending on whether the ray origin lies inside the bounding box
                    if (bBox.Contains(o)) { t[1] = tPlaneFar; t[0] = 0.0f; }
                    else { t[0] = tPlaneFar; }
                }
                // Nothing hit. 
                else
                {
                    node = nullptr;
                }
            }
        }

        template<typename LeafLambda/*, typename InnerLambda*/>
        __host__ __device__ void TestRay(const vec2& o, const vec2& d, LeafLambda onIntersectLeaf) const
        {           
            if (!m_isConstructed) { return; }
                
            BIH2DNearFar t;
            if(!TestRayBBox(o, d, m_bBox, t)) { return; }

            // Create a stack            
            BIH2DRayStackElement stack[kBIH2DStackSize];
            BBox2f bBox = m_bBox;
            BIH2DNode* node = &m_nodes[0];
            uint nodeIdx = 0;
            int stackIdx = -1;

            assert(node);            

            do
            {                
                // If there's no node in place, pop the stack
                if (!node)
                {
                    assert(stackIdx >= 0);
                    const BIH2DRayStackElement& element = stack[stackIdx--];
                    t = element.t;
                    bBox = element.bBox;
                    node = &m_nodes[element.nodeIdx];
                    assert(node);
                }

                // Node is a leaf?
                const uchar axis = node->GetAxis();
                if (axis == kBIHLeaf)
                {
                    if (node->IsValidLeaf())
                    {
                        t[0] = min(t[0], onIntersectLeaf(node->GetPrimIndex(), t[0]));
                    }
                    node = nullptr;
                }
                // ...or an inner node.
                else
                {                    
                    // Left-hand node is likely closer, so traverse that one first
                    if (o[axis] < node->planes[BIH2DNode::kLeft])
                    {
                        RayTraverseInnerNode(o, d, node, axis, 0, t, bBox, stack, stackIdx);
                    }
                    // ...otherwise traverse right-hand node first
                    else
                    {
                        RayTraverseInnerNode(o, d, node, axis, 1, t, bBox, stack, stackIdx);
                    }                    
                }
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

            t[0] = max(0.0f, cwiseMax(tNearPlane));
            t[1] = cwiseMin(tFarPlane);
            return t[0] < t[1];
        }

    protected:
        BIH2DNode*                  m_nodes;
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
            AssetHandle<Host::Vector<BIH2DNode>>    m_hostNodes;
            std::vector<uint>                       m_primitiveIdxs;
            BIH2DStats                              m_stats;

            Device::BIH2DAsset*                     cu_deviceInstance;
        };

    
    }
}