#pragma once

#include "Math.cuh"

namespace Enso
{
    // Constants for the Fowler-Noll-Vo hash function https://en.wikipedia.org/wiki/Fowler-Noll-Vo_hash_function
    constexpr uint kFNVPrime = 0x01000193u;
    constexpr uint kFNVOffset = 0x811c9dc5u;

    // Mix and combine hashes
    template<typename T>
    __host__ __device__ inline T HashCombine(const T& a, const T& b)
    {
        static_assert(std::is_integral<T>::value, "HashCombine requires integral type.");
        
        // My Shadertoy function
        return (((a << (31 - (b & 31))) | (a >> (b & 31)))) ^
                ((b << (a & 31)) | (b >> (31 - (a & 31))));

        // Based on Boost's hash_combine().
        // NOTE: Collides badly when hashing a string with only one letter difference
        //return b + 0x9e3779b9 + (a << 6) + (a >> 2);
    }

    __host__ __device__ inline uint HashCombine(const float& a, const float& b)
    {
        return HashCombine(*reinterpret_cast<const uint*>(&a), *reinterpret_cast<const uint*>(&b));
    }

    template<typename Type> __host__ __device__  inline uint HashCombine(const Type& a) { return a; }

    template<typename Type, typename... Pack>
    __host__ __device__ inline uint HashCombine(const Type& a, const Type& b, Pack... pack)
    {
        return HashCombine(HashCombine(a, b), HashCombine(pack...));
    }

    // Compute a 32-bit Fowler-Noll-Vo hash for the given input
    __host__ __device__ inline uint HashOf(const uint& i)
    {
        uint h = (kFNVOffset ^ (i & 0xffu)) * kFNVPrime;
        h = (h ^ ((i >> 8u) & 0xffu)) * kFNVPrime;
        h = (h ^ ((i >> 16u) & 0xffu)) * kFNVPrime;
        h = (h ^ ((i >> 24u) & 0xffu)) * kFNVPrime;
        return h;
    }

    __host__ __device__ inline uint HashOf(const float& i)
    {
        return HashOf(*reinterpret_cast<const uint*>(&i));
    }

    __host__ __device__ inline uint HashOf(const int& i)
    {
        return HashOf(*reinterpret_cast<const uint*>(&i));
    }

    template<typename Type, typename... Pack>
    __host__ __device__ inline uint HashOf(const Type& v0, const Pack&... pack)
    {
        return HashCombine(HashOf(v0), HashOf(pack...));
    }

    template<typename... Pack>
    __host__ __device__ inline float HashOfAsFloat(const Pack&... pack)
    {
        auto h = HashOf(pack...);
        return float(h) / float(0xffffffff);
    }

    // Compute a 32-bit Fowler-Noll-Vo hash for an array of bytes
    __host__ __device__ inline uint32_t HashOfArray(const char* data, const size_t numBytes)
    {
        uint32_t hash = 0x811c9dc5u;
        for (int i = 0; i < numBytes; ++i)
        {
            hash = (hash ^ data[i]) * 0x01000193u;
        }
        return hash;
    }
}