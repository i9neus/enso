#pragma once

#include "core/math/bbox/BBox2.cuh"
#include <map>

#include "core/Asset.cuh"
#include "core/Image.cuh"
#include "core/Vector.cuh"

namespace Enso
{
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

        __host__ __device__ BIH2DNodeBase() {}

        __host__ __device__ __forceinline__ uchar GetAxis() const { return uchar(flags & uint(3)); }
        __host__ __device__ __forceinline__ const uint* GetPrimIdxs() const { return data.primIdxs; }
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
            CudaAssertDebug(i < ~uint(3));
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
}