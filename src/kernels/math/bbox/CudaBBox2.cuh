#pragma once

#include "../CudaMath.cuh"
 
namespace Cuda
{
    template<typename VecType, typename RealType = float>
    struct BBox2
    {
    public:
        using ScalarType = typename VecType::kType;

        BBox2() noexcept : lower(ScalarType(0)), upper(ScalarType(0)) {}
        BBox2(const BBox2&) = default;
        BBox2(BBox2&&) = default;
        ~BBox2() = default;

        __forceinline__ BBox2(const VecType & l, const VecType & u) : lower(l), upper(u) noexcept {}
        __forceinline__ BBox2(const ScalarType & lx, const ScalarType & ly, const ScalarType & ux, const ScalarType & uy) : lower(lx, ly), upper(ux, uy) noexcept {}

        template<typename OtherType> __forceinline__  BBox2(const BBox2<OtherType>& other) :
            lower(OtherType(other.lower.x), OtherType(other.lower.y)), 
            upper(OtherType(other.upper.x), OtherType(other.upper.y)) {}

        __host__ __device__ __forceinline__ static BBox2 MakeInfinite() { return BBox2(-kFltMax, -kFltMax, kFltMax, kFltMax); }
        __host__ __device__ __forceinline__ static BBox2 MakeInvalid() { return BBox2(kFltMax, kFltMax, -kFltMax, -kFltMax); }
        __host__ __device__ __forceinline__ static BBox2 MakeZeroArea() { return BBox2(); }

        __host__ __device__ __forceinline__ bool HasZeroArea() const { return upper.x == lower.x || upper.y == lower.y; }
        __host__ __device__ __forceinline__ bool HasPositiveArea() const { return upper.x > lower.x && upper.y > lower.y; }
        __host__ __device__ __forceinline__ bool HasValidArea() const { return upper.x >= lower.x && upper.y >= lower.y; }
        __host__ __device__ __forceinline__ bool IsInfinite() const { return upper.x == kFltMax || lower.x == -kFltMax || upper.y == kFltMax || lower.y == -kFltMax; }
        __host__ __device__ __forceinline__ ScalarType Area() const { return (upper.x - lower.x) * (upper.y - lower.y); }
        __host__ __device__ __forceinline__ ScalarType Width() const { return upper.x - lower.x; }
        __host__ __device__ __forceinline__ ScalarType Height() const { return upper.y - lower.y; }
        __host__ __device__ __forceinline__ ScalarType EdgeLength(const int idx) const { return upper[idx] - lower[idx]; }
        __host__ __device__ __forceinline__ uint MaxAxis() const { return (Width() > Height()) ? 0 : 1; }
        __host__ __device__ __forceinline__ uint MinAxis() const { return (Width() < Height()) ? 0 : 1; }
        
        __host__ __device__ __forceinline__ VecType Centroid() const
        {
            return VecType(ScalarType(RealType(0.5) * RealType(upper[0] - lower[0])), 
                           ScalarType(RealType(0.5) * RealType(upper[1] - lower[1])));
        }

        __host__ __device__ __forceinline__ ScalarType Centroid(const uint axis) const
        {
            return ScalarType(RealType(0.5) * RealType(upper[axis] - lower[axis]));
        }

        __host__ __device__ __forceinline__ ScalarType operator[](const int idx) { return v[idx]; }

        __host__ __device__ __forceinline__ bool Contains(const VecType & p)
        {
            return p.x >= lower.x && p.x <= upper.x && p.y >= lower.y && p.y <= upper.y;
        }

        __host__ __device__ __forceinline__ bool Contains(const BBox2 & other)
        {
            return lower.x <= other.lower.x && lower.y <= other.lower.y && upper.x >= other.upper.x && upper.y >= other.upper.y;
        }

        __host__ __device__ __forceinline__ void Grow(const ScalarType & ammount)
        {
            lower.x -= ammount; lower.y -= ammount;
            upper.x += ammount; upper.y += ammount;
        }

        __host__ __device__ __forceinline__ void Grow(const VecType & ammount)
        {
            lower.x -= ammount.x; lower.y -= ammount.y;
            upper.x += ammount.x; upper.y += ammount.y;
        }

    public:
        union
        {
            VecType v[2];
            struct
            {
                VecType lower;
                VecType upper;
            };
        };
    };

    using BBox2f = BBox2<vec2>;
    using BBox2i = BBox2<ivec2>;
    using BBox2u = BBox2<uvec2>;

    template<typename T>
    __host__ __device__ __forceinline__ BBox2<T> Union(const BBox2<T>& a, const BBox2<T>& b)
    {
        return BBox2<T>(min(a.lower.x, b.lower.x), min(a.upper.y, b.upper.y), max(a.upper.x, b.upper.x), max(a.upper.y, b.upper.y));
    }

    template<typename T>
    __host__ __device__ __forceinline__ BBox2<T> Intersection(const BBox2<T>& a, const BBox2<T>& b)
    {
        return BBox2<T>(max(a.lower.x, b.lower.x), max(a.upper.y, b.upper.y), min(a.upper.x, b.upper.x), min(a.upper.y, b.upper.y));
    }

    template<typename T>
    __host__ __device__ __forceinline__ bool operator==(const BBox2<T>& a, const BBox2<T>& b)
    {
        return a.lower.x == b.lower.x && a.lower.y == b.lower.y && a.upper.x == b.upper.x && a.upper.y == b.upper.y;
    }

    template<typename T>
    __host__ __device__ __forceinline__ bool operator!=(const BBox2<T>& a, const BBox2<T>& b)
    {
        return !(a == b);
    }

    template<typename T>
    __host__ __device__ __forceinline__ BBox2<T>  operator&(const BBox2<T>& a, const BBox2<T>& b)
    {
        return Intersection(a, b);
    }

    template<typename T>
    __host__ __device__ __forceinline__ BBox2<T>  operator|(const BBox2<T>& a, const BBox2<T>& b)
    {
        return Union(a, b);
    }
}