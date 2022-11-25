#pragma once

#include "CudaMath.cuh"

namespace Cuda
{
    class Complex : public vec2
    {
    public:
        __host__ __device__ Complex() : vec2(0.0f) {}
        __host__ __device__ Complex(const float r, const float i) : vec2(r, i) {}
        __host__ __device__ Complex(const vec2& v) : vec2(v) {}

        __host__ __device__ __forceinline__ Complex& operator =(const vec2& rhs) 
        { 
            static_cast<vec2&>(*this) = rhs;
            return *this; 
        }

        __host__ __device__ __forceinline__ Complex& operator *=(const Complex& rhs)
        {
            *this = Complex(x * rhs.x - y * rhs.y, x * rhs.y + y * rhs.x);
            return *this;
        }
        __host__ __device__ __forceinline__ Complex& operator +=(const Complex& rhs)
        {
            x += rhs.x; y += rhs.y;
            return *this;
        }

        __host__ __device__ __forceinline__ Complex& operator -=(const Complex& rhs)
        {
            x -= rhs.x; y -= rhs.y;
            return *this;
        }

        __host__ __device__ __forceinline__ float& Real() { return x; }
        __host__ __device__ __forceinline__ const float& Real() const { return x; }
        __host__ __device__ __forceinline__ float& Img() { return y; }
        __host__ __device__ __forceinline__ const float& Img() const { return y; }

        __host__ __device__ __forceinline__ float Magnitude() const { return length(*this); }
        __host__ __device__ __forceinline__ float Magnitude2() const { return length2(*this); }

        __host__ __device__ __forceinline__ Complex operator+(Complex& rhs) const { return Complex(x + rhs.x, y + rhs.y); }
        __host__ __device__ __forceinline__ Complex operator-(Complex& rhs) const { return Complex(x - rhs.x, y - rhs.y); }
        __host__ __device__ __forceinline__ Complex operator*(Complex& rhs) const { return Complex(x * rhs.x - y * rhs.y, x * rhs.y + y * rhs.x); }
    };

    __host__ __device__ __forceinline__ Complex Conjugate(const Complex& other) { return Complex(other.x, fabsf(other.y)); }
}