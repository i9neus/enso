#pragma once

#include "core/math/bbox/BBox2.cuh"
#include "core/assets/Asset.cuh"
#include "core/containers/Vector.cuh"
#include "core/2d/primitives/GenericIntersector.cuh"
#include <map>

namespace Enso
{
    namespace BIH2D
    {
        template<typename NodeType> class Builder;

        enum BIHFlags : unsigned char
        {
            kBIHX = 0,
            kBIHY = 1,
            kBIHLeaf = 2
        };

        struct NodeDataFull
        {
            float        planes[2];
            uint         primIdxs[2];
        };

        struct NodeDataCompact
        {
            union
            {
                float        planes[2];
                uint         primIdxs[2];
            };
        };

        template<typename NodeDataType>
        struct NodeBase
        {
        public:
            enum _attrs : uint { kLeft = 0, kRight = 1, kInvalidLeaf = 0xffffffff };

            __host__ __device__ NodeBase() {}

            __host__ __device__ __forceinline__ uchar GetAxis() const { return uchar(flags & uint(3)); }
            __host__ __device__ __forceinline__ const uint* GetPrimIdxs() const { return data.primIdxs; }
            __host__ __device__ __forceinline__ uint GetChildIndex() const { return flags >> 2; }
            __host__ __device__ __forceinline__ bool IsValidLeaf() const { return data.primIdxs[0] != kInvalidLeaf; }
            __host__ __device__ __forceinline__ bool IsLeaf() const { return uchar(flags & uint(3)) == kBIHLeaf; }

            __host__ __device__ __forceinline__ BBox2f GetLeftBBox(BBox2f parentBBox) const
            {
                parentBBox.upper[flags & 3u] = data.planes[NodeBase::kLeft];
                return parentBBox;
            }

            __host__ __device__ __forceinline__ BBox2f GetRightBBox(BBox2f parentBBox) const
            {
                parentBBox.lower[flags & 3u] = data.planes[NodeBase::kRight];
                return parentBBox;
            }

            __host__ __device__ __forceinline__ void MakeInner(const uint& i, const uint& split, const float& left, const float& right,
                const uint& primIdxStart, const uint& primIdxEnd)
            {
                CudaAssertDebug(i < ~uint(3));
                flags = (i << 2) | (split & uint(3));
                data.planes[NodeBase::kLeft] = left;
                data.planes[NodeBase::kRight] = right;

                // If we're using full nodes in this tree, include the start and end indices in the inner node
                if (std::is_same<NodeDataType, NodeDataFull>::value)
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

            __host__ __device__ __forceinline__ void MakeInvalidLeaf()
            {
                flags = kBIHLeaf;
                data.primIdxs[0] = kInvalidLeaf;
                data.primIdxs[1] = kInvalidLeaf;
            }

        public:
            uint         flags;
            NodeDataType data;
        };

        using CompactNode = NodeBase<NodeDataCompact>;
        using FullNode = NodeBase<NodeDataFull>;

        template<typename NodeDataType>
        struct BIHData
        {
            __device__ void Validate() const
            {
                CudaAssert(nodes);
                CudaAssert(indices);
            }

            bool                        isConstructed = false;
            bool                        testAsList = false;
            BBox2f                      bBox;
            NodeDataType* nodes = nullptr;
            uint* indices = nullptr;
            uint                        numNodes = 0;
            uint                        numPrims = 0;
            int                         treeDepth = -1;
        };

        struct Stats
        {
            float buildTime = 0.f;
            uchar maxDepth = 0;
            uint numInnerNodes = 0;
            uint numLeafNodes = 0;
        };
    }
}