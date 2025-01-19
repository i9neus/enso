#pragma once

#include "../BIHData.cuh"
#include "core/2d/primitives/GenericIntersector.cuh"

namespace Enso
{
    namespace BIH2D
    {
        namespace Traverser
        {
            enum : int
            {
                kStackSize = 10,
                kMinPrimsInTree = 5
            };

            struct StackElement
            {
                BBox2f      bBox;
                uint        nodeIdx;
                uchar       depth;
            };
            
            using Stack = StackElement[kStackSize];

            template<typename InnerLambda>
            __host__ __device__ inline static void OnPrimitiveIntersectInner(const BBox2f& bBox, const uchar& depth, InnerLambda onIntersectInner) { onIntersectInner(bBox, depth); }
            template<>
            __host__ __device__ inline static void OnPrimitiveIntersectInner(const BBox2f&, const uchar&, nullptr_t) { }
        };
    }
}