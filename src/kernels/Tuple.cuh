#pragma once

#include "CudaCommonIncludes.cuh"

namespace
{
    struct TupleEndType {};

    template<int TDepth, typename TType, typename... TTypePack>
    class TupleImpl : public TupleImpl<TDepth - 1, TTypePack...>
    {
    public:
        enum _attrs : int { Depth = TDepth };
        using Type = TType;
        using Super = TupleImpl<Depth - 1, TTypePack...>;

        TupleImpl() {}

        template<typename Type, typename... Args>
        TupleImpl(const Type& data, Args... args) : Super(args...), m_element(data) {}

        __host__ __device__ Type& GetElement() { return m_element; }
        __host__ __device__ const Type& GetElement() const { return m_element; }

    private:
        Type m_element;
    };

    template<typename... TypePack>
    class TupleImpl<-1, TupleEndType, TypePack...> { };

    template<int Index, typename TupleType, typename = typename std::enable_if<Index == TupleType::Depth>::type >
    __host__ __device__ typename TupleType::Type& GetImpl(TupleType& tuple)
    {
        return tuple.GetElement();
    }

    template<int Index, typename TupleType, typename = typename std::enable_if<Index != TupleType::Depth>::type>
    __host__ __device__ decltype(GetImpl<Index>(std::declval<typename TupleType::Super&>())) GetImpl(TupleType& tuple)
    {
        return GetImpl<Index>(static_cast<typename TupleType::Super&>(tuple));
    }

    template<int Index, typename TupleType, typename = typename std::enable_if<Index == TupleType::Depth>::type >
    __host__ __device__ typename const TupleType::Type& GetImpl(const TupleType& tuple)
    {
        return tuple.GetElement();
    }

    template<int Index, typename TupleType, typename = typename std::enable_if<Index != TupleType::Depth>::type>
    __host__ __device__ decltype(GetImpl<Index>(std::declval<const typename TupleType::Super&>())) GetImpl(const TupleType& tuple)
    {
        return GetImpl<Index>(static_cast<const typename TupleType::Super&>(tuple));
    }
}

namespace Cuda
{
    template<typename... TypePack>
    class Tuple : public TupleImpl<sizeof...(TypePack) - 1, TypePack..., TupleEndType>
    {
    public:
        using Super = TupleImpl<sizeof...(TypePack) - 1, TypePack..., TupleEndType>;

        Tuple() { }

        template<typename... Args, typename = typename std::enable_if<sizeof...(Args) == sizeof...(TypePack)>::type>
        Tuple(Args... args) : Super(args...) {}
    };

    template<int Index, typename TupleType>
    __host__ __device__ decltype(GetImpl<Index>(std::declval<typename TupleType::Super&>())) Get(TupleType& tuple)
    {
        return GetImpl<Index>(tuple);
    }

    template<int Index, typename TupleType>
    __host__ __device__ decltype(GetImpl<Index>(std::declval<const typename TupleType::Super&>())) Get(const TupleType& tuple)
    {
        return GetImpl<Index>(tuple);
    }
}