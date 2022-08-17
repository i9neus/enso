#pragma once

#include "../CudaMath.cuh"
 
namespace Cuda
{
    template<typename T>
    struct BBox2
    {
    public:
        using ScalarType = typename T::kType;
        using VecType = T;

        __host__ __device__ BBox2() noexcept : lower(ScalarType(0)), upper(ScalarType(0)) {}
        __host__ __device__ BBox2(const BBox2&) = default;
        __host__ __device__ BBox2(BBox2&&) = default;
        __host__ __device__ BBox2& operator=(const BBox2&) = default;
        __host__ __device__ ~BBox2() = default;

        __host__ __device__ __forceinline__ BBox2(const VecType & l, const VecType & u) noexcept : lower(l), upper(u) {}
        __host__ __device__ __forceinline__ BBox2(const ScalarType & lx, const ScalarType & ly, const ScalarType & ux, const ScalarType & uy) noexcept : lower(lx, ly), upper(ux, uy) {}

        template<typename OtherType>  
        __host__ __device__ __forceinline__  BBox2(const BBox2<OtherType>& other) :
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
            return VecType(ScalarType(ScalarType(0.5) * ScalarType(upper[0] + lower[0])), 
                           ScalarType(ScalarType(0.5) * ScalarType(upper[1] + lower[1])));
        }

        __host__ __device__ __forceinline__ VecType Dimensions() const
        {
            return VecType(upper[0] - lower[0], upper[1] - lower[1]);
        }

        __host__ __device__ __forceinline__ ScalarType Centroid(const uint axis) const
        {
            return ScalarType(ScalarType(0.5) * ScalarType(upper[axis] + lower[axis]));
        }

        __host__ __device__ __forceinline__ VecType& operator[](const int idx) { return v[idx]; }
        __host__ __device__ __forceinline__ const VecType& operator[](const int idx) const { return v[idx]; }

        __host__ __device__ __forceinline__ bool Contains(const VecType & p) const
        {
            return p.x >= lower.x && p.x <= upper.x && p.y >= lower.y && p.y <= upper.y;
        }

        __host__ __device__ __forceinline__ bool Contains(const BBox2 & other) const
        {
            return lower.x <= other.lower.x && lower.y <= other.lower.y && upper.x >= other.upper.x && upper.y >= other.upper.y;
        }

        __host__ __device__ __forceinline__ bool Intersects(const VecType& p) const
        {
            return p.x >= lower.x && p.x <= upper.x && p.y >= lower.y && p.y <= upper.y;
        }

        __host__ __device__ __forceinline__ bool Intersects(const BBox2& other) const
        {
            return max(lower.x, other.lower.x) < min(upper.x, other.upper.x) && max(lower.y, other.lower.y) < min(upper.y, other.upper.y);
        }

        __host__ __device__ __forceinline__ void Grow(const ScalarType & ammount)
        {
            lower.x -= ammount; lower.y -= ammount;
            upper.x += ammount; upper.y += ammount;
        }

        // Grows the bounding box by a vector accmount
        __host__ __device__ __forceinline__ void Grow(const VecType & ammount)
        {
            lower.x -= ammount.x; lower.y -= ammount.y;
            upper.x += ammount.x; upper.y += ammount.y;
        }

        // Checked to see whether lower < upper and inverts them if not
        __host__ __device__ __forceinline__ void Rectify()
        {
            if (lower.x > upper.x) { swap(lower.x, upper.x); }
            if (lower.y > upper.y) { swap(lower.y, upper.y); }      
        }

        __host__ __device__ __forceinline__ BBox2& operator*=(const ScalarType& scale)
        {
            const VecType centroid = Centroid();
            const VecType dimensions = Dimensions();
            lower = centroid - dimensions * ScalarType(0.5) * scale;
            upper = centroid + dimensions * ScalarType(0.5) * scale;
            return *this;
        }

        __host__ std::string& Format() const
        {
            return tfm::format("{%s, %s}", lower.format(), upper.format());
        }

        __host__ __device__ __forceinline__ bool PointOnPerimiter(const vec2& p, float thickness)
        {
            thickness *= 0.5f;
            return (p.x >= lower.x - thickness && p.y >= lower.y - thickness && p.x <= upper.x + thickness && p.y <= upper.y + thickness) &&
                (p.x <= lower.x + thickness || p.y <= lower.y + thickness || p.x >= upper.x - thickness || p.y >= upper.y - thickness);
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
        return BBox2<T>(min(a.lower.x, b.lower.x), min(a.lower.y, b.lower.y), max(a.upper.x, b.upper.x), max(a.upper.y, b.upper.y));
    }

    template<typename T>
    __host__ __device__ __forceinline__ BBox2<T> Union(const BBox2<T>& a, const vec2& b)
    {
        return BBox2<T>(min(a.lower.x, b.x), min(a.lower.y, b.y), max(a.upper.x, b.x), max(a.upper.y, b.y));
    }

    template<typename T>
    __host__ __device__ __forceinline__ BBox2<T> Intersection(const BBox2<T>& a, const BBox2<T>& b)
    {
        return BBox2<T>(max(a.lower.x, b.lower.x), max(a.lower.y, b.lower.y), min(a.upper.x, b.upper.x), min(a.upper.y, b.upper.y));
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
    __host__ __device__ __forceinline__ BBox2<T> Grow(const BBox2<T>& bBox, const typename BBox2<T>::ScalarType& ammount)
    {
        return BBox2<T>(bBox.lower.x - ammount, bBox.lower.y - ammount, bBox.upper.x + ammount, bBox.upper.y + ammount);
    }

    template<typename VecType>
    __host__ __device__ __forceinline__ BBox2<VecType> Grow(const BBox2<VecType>& bBox, const VecType& ammount)
    {
        return BBox2<VecType>(bBox.lower.x - ammount.x, bBox.lower.y - ammount.y, bBox.upper.x + ammount.x, bBox.upper.y + ammount.y);
    }

    template<typename VecType>
    __host__ __device__ __forceinline__ BBox2<VecType> Rectify(const BBox2<VecType>& bBox)
    {
        BBox2<VecType> rect(bBox);
        rect.Rectify();
        return rect;
    }
}