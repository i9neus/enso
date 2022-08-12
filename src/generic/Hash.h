#pragma once

#include "StdIncludes.h"

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

size_t __forceinline__ _Hash_bytes(const void* ptr, size_t len, size_t seed)
{
    const size_t m = 0x5bd1e995;
    size_t hash = seed ^ len;
    const char* buf = static_cast<const char*>(ptr);

    // Mix 4 bytes at a time into the hash.
    while (len >= 4)
    {
        size_t k = *buf;
        k *= m;
        k ^= k >> 24;
        k *= m;
        hash *= m;
        hash ^= k;
        buf += 4;
        len -= 4;
    }

    // Handle the last few bytes of the input array.
    switch (len)
    {
    case 3:
        hash ^= static_cast<unsigned char>(buf[2]) << 16;
    case 2:
        hash ^= static_cast<unsigned char>(buf[1]) << 8;
    case 1:
        hash ^= static_cast<unsigned char>(buf[0]);
        hash *= m;
    };

    // Do a few final mixes of the hash.
    hash ^= hash >> 13;
    hash *= m;
    hash ^= hash >> 15;
    return hash;
}