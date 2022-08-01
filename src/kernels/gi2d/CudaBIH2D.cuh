#pragma once

#include "generic/StdIncludes.h"
#include "kernels/math/bbox/CudaBBox2.cuh"
#include <map>

#include "../CudaAsset.cuh"
#include "../CudaManagedObject.cuh"
#include "../CudaImage.cuh"
#include "../CudaManagedArray.cuh"

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
        __host__ __device__ BIH2DNode() = default;

        __host__ __device__ __forceinline__ uchar GetAxis() const { return uchar(data & 3u); }
        __host__ __device__ __forceinline__ uint GetPrimIndex() const { return data >> 2; }
        __host__ __device__ __forceinline__ uint GetChildIndex() const { return data >> 2; }
        
        __host__ __device__ __forceinline__ void SetData(const uint idx, const uint flags) 
        { 
            assert(idx < ~3u);
            data = (idx << 2) | (flags & 3u);
        }
        
        uint            data;
        float           left;
        float           right;
    };

    

    namespace Device
    {
        class BIH2D : public Asset
        {
        private:
            struct StackElement
            {
                BBox2f      bBox;
                uint        nodeIdx;
            };

        public:
            __device__ BIH2D() : m_isConstructed(false) {}
            __device__ ~BIH2D() = default;

            __device__ bool IsConstructed() const { return m_isConstructed; }

            template<typename InteresectLambda>
            __device__ bool TestPoint(const vec2& p, InteresectLambda onIntersect) const
            {
                #define kBIHStackSize 10
                StackElement stack[kBIHStackSize];                
                BBox2f bBoxA = m_bBox, bBoxB;
                BIH2DNode* node = &m_nodes[0];                
                uint nodeIdx = 0;
                int stackIdx = -1;
                
                do
                {
                    // If there's no node in place, pop the stack
                    if (!node)
                    {
                        nodeIdx = stack[stackIdx].nodeIdx;
                        bBoxA = stack[stackIdx--].bBox;
                        node = &m_nodes[nodeIdx];
                        assert(node);
                    }

                    // Node is a leaf?
                    const uchar axis = node->GetAxis();
                    if (axis == kBIHLeaf)
                    {
                        onIntersect(node->GetPrimIndex());
                    }
                    // ...or an inner node.
                    else
                    {
                        bBoxA = parentBBox;
                        bBoxA[1][axis] = node->left;
                        bBoxB = parentBBox;
                        bBoxB[0][axis] = node->right;

                        // Left box hit?
                        if (bBoxA.Contains(p))
                        {
                            // ...and right box hit?
                            if (bBoxB.Contains(p) && stackIdx < kBIHStackSize)
                            {
                                stack[stackIdx++] = { bBoxB, node->GetChildIndex() + 1 };
                            }

                            nodeIdx = node->GetChildIndex();
                            node = &m_nodes[nodeIdx];
                        }
                        // Right box hit?
                        else if (bBoxB.Contains(p))
                        {
                            nodeIdx = node->GetChildIndex() + 1;
                            node = &m_nodes[nodeIdx];
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

            __device__ void Synchronise(BIH2DNode* nodes, const BBox2f& bBox)
            {
                m_nodes = nodes;
                m_bBox = bBox;
                m_isConstructed = true;
            }       

        private:
            BIH2DNode*          m_nodes;
            BBox2f              m_bBox;
            bool                m_isConstructed;
        };
    }

    namespace Host
    {
        class BIH2D : public Asset
        {
        public:
            friend class BIH2DBuilder;

            __host__ BIH2D(const std::string& id);
            __host__ virtual ~BIH2D();

            __host__ virtual void OnDestroyAsset() override final;

        private:
            __host__ void Resize(const size_t numPrimitives);

            AssetHandle<Host::Array<BIH2DNode>>         m_hostNodes;
            BBox2f                                      m_bBox;
            bool                                        m_isConstructed;
            Device::BIH2D*                              cu_deviceData;
        };        

        class BIH2DBuilder
        {
        public:
            BIH2DBuilder(Host::BIH2D& bih) : 
                m_bih(bih), 
                m_hostNodes(bih.m_hostNodes)
            {}

            template<typename GetBBoxLambda>
            void Build(std::vector<uint>& primitives, GetBBoxLambda getPrimitiveBBox)
            {                               
                Assert(m_hostNodes);
                m_bih.Resize(primitives.size());
                
                Timer timer;
                
                // Find the global bounding box
                m_bih.m_bBox = BBox2f::MakeInvalid();

                AssertMsgFmt(m_bih.m_bBox.HasValidArea() && !m_bih.m_bBox.IsInfinite(),
                    "BIH bounding box is invalid: {%s, %s}", m_bih.m_bBox[0].format().c_str(), m_bih.m_bBox[1].format().c_str());

                for (const auto idx : primitives)
                {
                    const BBox2f primBBox = getPrimitiveBBox(idx);
                    AssertMsgFmt(primBBox.HasValidArea(),
                        "BIH primitive at index %i has returned an invalid bounding box: {%s, %s}", primBBox[0].format().c_str(), primBBox[1].format().c_str());
                    m_bih.m_bBox = Union(m_bih.m_bBox, primBBox);
                }

                // Construct the bounding interval hierarchy
                uint nodeListIdx = 1;
                uchar maxDepth = 0;
                BuildPartition(primitives, 0, primitives.size(), 0, nodeListIdx, 0, maxDepth, m_bih.m_bBox, getPrimitiveBBox);

                m_bih.m_isConstructed = true;
                Log::Write("Constructed BIH in %.1fms", timer.Get() * 1e-3f);
            }

        private:
            template<typename GetBBoxLambda>
            void BuildPartition(std::vector<uint>& primitives, const int i0, const int i1, const uint thisIdx, uint& nodeListIdx, 
                                const uchar depth, uchar& maxDepth, const BBox2f& parentBBox, GetBBoxLambda getPrimitiveBBox)
            {
                // Sanity checks
                Assert(depth < 32); 
                Assert(i0 != i1);

                BIH2DNode& node = (*m_hostNodes)[thisIdx];
             
                // If this node only contains one primitive, it's a leaf
                if (i1 == i0 + 1)
                {                                        
                    node.SetData(primitives[i0], kBIHLeaf);
                    return;
                }

                const uint axis = parentBBox.MaxAxis();
                const float split = parentBBox.Centroid(axis);
                BBox2f leftBBox = parentBBox, rightBBox = parentBBox;
                leftBBox[1][axis] = -kFltMax;
                rightBBox[0][axis] = kFltMax;

                // Sort the primitives along the dominant axis
                int j = 0;
                uint sw;
                for (int i = i0; i < i1; ++i)
                {
                    const BBox2f primBBox = getPrimitiveBBox(primitives[i]);
                    const vec2 centroid = primBBox.Centroid();

                    if (centroid[axis] < split)
                    {
                        // Update the partition position
                        leftBBox[1][axis] = max(leftBBox[1][axis], primBBox[1][axis]);
                        // Swap the element into the left-hand partition
                        sw = primitives[j]; primitives[j] = primitives[i]; primitives[i] = sw;
                        // Increment the partition index
                        ++j;
                    }
                    else
                    {
                        // Update the partition position
                        rightBBox[0][axis] = min(rightBBox[0][axis], primBBox[0][axis]);
                    }
                }

                // Increment the list index to "allocate" two new nodes
                nodeListIdx += 2;

                // Build the inner node          
                node.SetData(nodeListIdx - 2, axis);
                node.left = leftBBox[1][axis];
                node.right = rightBBox[0][axis];

                // Build the child nodes
                BuildPartition(primitives, i0, j, nodeListIdx - 2, nodeListIdx, leftBBox, getPrimitiveBBox);
                BuildPartition(primitives, j, i1, nodeListIdx - 1, nodeListIdx, rightBBox, getPrimitiveBBox);
            }

        private:
            Host::BIH2D&                                m_bih;
            AssetHandle<Host::Array<BIH2DNode>>         m_hostNodes;
        };
    }

}