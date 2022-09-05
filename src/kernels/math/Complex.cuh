#pragma once

#include "CudaMath.cuh"

namespace Cuda
{
    class Complex
    {
    public:
        __host__ __device__ Complex() {}
        __host__ __device__ Complex(const float r, const float i) : m_data(r, i) {}

        __host__ __device__ __forceinline__ Complex& operator *=(const Complex& rhs)
        {
            Complex(m_data.x * rhs.m_data.x + m_data.y * rhs.m_data.y, m_data.x * rhs.m_data.y + m_data.y * rhs.m_data.x);
            return *this;
        }

        __host__ __device__ __forceinline__ Complex& operator +=(const Complex& rhs)
        {
            m_data.x += rhs.m_data.x; m_data.y += rhs.m_data.y;
            return *this;
        }

        __host__ __device__ __forceinline__ Complex& operator -=(const Complex& rhs)
        {
            m_data.x -= rhs.m_data.x; m_data.y -= rhs.m_data.y;
            return *this;
        }

        __host__ __device__ __forceinline__ float& Real() { return m_data.x; }
        __host__ __device__ __forceinline__ const float& Real() const { return m_data.x; }
        __host__ __device__ __forceinline__ float& Img() { return m_data.y; }
        __host__ __device__ __forceinline__ const float& Img() const { return m_data.y; }

        __host__ __device__ __forceinline__ Complex operator+(Complex& rhs) const { return Complex(m_data.x + rhs.m_data.x, m_data.y + rhs.m_data.y); }
        __host__ __device__ __forceinline__ Complex operator-(Complex& rhs) const { return Complex(m_data.x - rhs.m_data.x, m_data.y - rhs.m_data.y); }
        __host__ __device__ __forceinline__ Complex operator*(Complex& rhs) const { return Complex(m_data.x * rhs.m_data.x + m_data.y * rhs.m_data.y, m_data.x * rhs.m_data.y + m_data.y * rhs.m_data.x); }

    private:
        vec2 m_data;
    };

    __host__ __device__ __forceinline__ Complex Conjugate(const Complex& other) { return Complex(other.Real(), fabsf(other.Img())); }
}