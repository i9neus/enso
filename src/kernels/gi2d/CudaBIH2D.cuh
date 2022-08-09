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
        enum _attrs : uint { kInvalidLeaf = 0xffffffff };
        
        __host__ __device__ BIH2DNode() = default;

        __host__ __device__ __forceinline__ uchar GetAxis() const { return uchar(data & 3u); }
        __host__ __device__ __forceinline__ uint GetPrimIndex() const { return idx; }
        __host__ __device__ __forceinline__ uint GetChildIndex() const { return data >> 2; }
        __host__ __device__ __forceinline__ bool IsValidLeaf() const { return idx != kInvalidLeaf; }
        __host__ __device__ __forceinline__ bool IsLeaf() const { return uchar(data & 3u) == kBIHLeaf; }

        __host__ __device__ __forceinline__ BBox2f GetLeftBBox(BBox2f parentBBox) const
        {
            parentBBox.upper[data & 3u] = left;
            return parentBBox;
        }

        __host__ __device__ __forceinline__ BBox2f GetRightBBox(BBox2f parentBBox) const
        {
            parentBBox.lower[data & 3u] = right;
            return parentBBox;
        }
        
        __host__ __device__ __forceinline__ void MakeInner(const uint& i, const uint& split, const float& l, const float& r) 
        { 
            assert(i < ~3u);
            data = (i << 2) | (split & 3u);
            left = l;
            right = r;
        }

        __host__ __device__ __forceinline__ void MakeLeaf(const uint& i)
        {
            data = kBIHLeaf;
            idx = i;
        }

        uint            data;
        union
        {
            struct
            {
                float           left;
                float           right;
            };
            uint                idx;
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

    namespace Host { class BIH2DBuilder; }

    class BIH2D
    {
    private:
        struct StackElement
        {
            BBox2f      bBox;
            uint        nodeIdx;
        };

    public:
        friend class Host::BIH2DBuilder;

        __host__ __device__ BIH2D();
        __host__ __device__ ~BIH2D() {};

        __host__ __device__ bool IsConstructed() const { return m_isConstructed; }

        template<typename LeafLambda/*, typename InnerLambda*/>
        __host__ __device__ void TestPoint(const vec2& p, LeafLambda onIntersectLeaf/*, InnerLambda onIntersectInner*/) const
        {
            if (!m_isConstructed) { return; }

            #define kBIHStackSize 10
            StackElement stack[kBIHStackSize];
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
                    nodeIdx = stack[stackIdx].nodeIdx;
                    bBox = stack[stackIdx--].bBox;
                    node = &m_nodes[nodeIdx];
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
                    /*if (kThreadIdx == 0)
                    {
                        printf("%i\n", iterIdx);
                        printf("A: {{%f, %f}, {%f, %f}}\n", bBoxA[0][0], bBoxA[0][1], bBoxA[1][0], bBoxA[1][1]);
                        //printf("B: {{%f, %f}, {%f, %f}}\n", bBoxB[0][0], bBoxB[0][1], bBoxB[1][0], bBoxB[1][1]);
                        printf("\n");
                    }*/
                    
                    //if (onIntersectInner) { onIntersectInner(bBox, depth); }

                    // Left box hit?
                    if(p[axis] < node->left)
                    {
                        // ...and right box hit?
                        if (p[axis] > node->right && stackIdx < kBIHStackSize)
                        {
                            stack[++stackIdx] = { bBox, node->GetChildIndex() + 1 };
                            stack[stackIdx].bBox[0][axis] = node->right;
                        }

                        nodeIdx = node->GetChildIndex();
                        node = &m_nodes[nodeIdx];
                        bBox[1][axis] = node->left;
                    }
                    // Right box hit?
                    else if (p[axis] > node->right)
                    {
                        nodeIdx = node->GetChildIndex() + 1;
                        node = &m_nodes[nodeIdx];
                        bBox[0][axis] = node->right;
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

        __host__ __device__ __forceinline__ const BBox2f&       GetBBox() const { return m_bBox; }
        __host__ __device__ __forceinline__ const BIH2DNode*    GetNodes() const { return m_nodes; }

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