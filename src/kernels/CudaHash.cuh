#pragma once

#include "math/CudaMath.cuh"

namespace Cuda
{
    // Constants for the Fowler-Noll-Vo hash function https://en.wikipedia.org/wiki/Fowler-Noll-Vo_hash_function
    constexpr uint kFNVPrime = 0x01000193u;
    constexpr uint kFNVOffset = 0x811c9dc5u;

    // Compute a 32-bit Fowler-Noll-Vo hash for the given input
    __host__ __device__ __forceinline__ uint HashOf(const uint i)
    {
        uint h = (kFNVOffset ^ (i & 0xffu)) * kFNVPrime;
        h = (h ^ ((i >> 8u) & 0xffu)) * kFNVPrime;
        h = (h ^ ((i >> 16u) & 0xffu)) * kFNVPrime;
        h = (h ^ ((i >> 24u) & 0xffu)) * kFNVPrime;
        return h;
    }

    // Mix and combine two hashes
    __host__ __device__ __forceinline__ uint HashCombine(uint a, uint b)
    {
        return (((a << (31u - (b & 31u))) | (a >> (b & 31u)))) ^
            ((b << (a & 31u)) | (b >> (31u - (a & 31u))));
    }

    template<typename... Vars>
    __host__ __device__ __forceinline__ uint HashOf(const uint& v0, const Vars&... var)
    {
        return HashCombine(HashOf(v0), HashOf(var...));
    }
}