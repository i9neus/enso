#pragma once

#include "math/CudaMath.cuh"

namespace Cuda
{
    // Constants for the Fowler-Noll-Vo hash function https://en.wikipedia.org/wiki/Fowler-Noll-Vo_hash_function
    constexpr uint kFNVPrime = 0x01000193u;
    constexpr uint kFNVOffset = 0x811c9dc5u;
    
    // Compute a 32-bit Fowler-Noll-Vo hash for the given input
    __device__ inline uint hashOf(const uint i)
    {
        uint h = (kFNVOffset ^ (i & 0xffu)) * kFNVPrime;
        h = (h ^ ((i >> 8u) & 0xffu)) * kFNVPrime;
        h = (h ^ ((i >> 16u) & 0xffu)) * kFNVPrime;
        h = (h ^ ((i >> 24u) & 0xffu)) * kFNVPrime;
        return h;
    }

    // Mix and combine two hashes
    __device__ inline uint hashCombine(uint a, uint b)
    {
        return (((a << (31u - (b & 31u))) | (a >> (b & 31u)))) ^
            ((b << (a & 31u)) | (b >> (31u - (a & 31u))));
    }

    template<typename... Vars>
    __device__ inline uint hashOf(const uint& v0, const Vars&... var)
    {
        return hashCombine(hashOf(v0), hashOf(var...));
    }

    /*__device__ inline uint hashCombine(uint a, uint b, uint c) { return hashCombine(hashCombine(a, b), c); }
    __device__ inline uint hashCombine(uint a, uint b, uint c, uint d) { return hashCombine(hashCombine(hashCombine(a, b), c), d); }
    __device__ inline uint hashCombine(uint a, uint b, uint c, uint d, uint e) { return hashCombine(hashCombine(hashCombine(hashCombine(a, b), c), d), e); }*/
}