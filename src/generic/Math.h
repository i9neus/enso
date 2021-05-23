#pragma once

#include <stdint.h>
#include <limits>
#include <math.h>

// Constants
constexpr float kPi = 3.14159265359f;
constexpr float kTwoPi = 2 * kPi;
constexpr float kFourPi = 4 * kPi;
constexpr float kHalfPi = 0.5 * kPi;
constexpr float kRoot2 = 1.41421356237f;
constexpr float kFltMax = 3.402823466e+38f;
constexpr float kPhi = 1.6180339887498948482045868343f;
constexpr float kInvPhi = 1 / kPhi;
constexpr float kLog2 = 0.6931471805599453f;

#undef min
#undef max

namespace math
{
    template<typename T> inline constexpr T mod2(const T& a, const T& b)    { return ((a % b) + b) % b; }
    template<typename T> inline constexpr T sqr(const T& x)                 { return x * x; }
    template<typename T> inline constexpr T cub(const T& x)                 { return x * x * x; }
    template<typename T> inline T min(const T& a, const T& b)               { return (a < b) ? a : b; }
    template<typename T> inline T max(const T& a, const T& b)               { return (a > b) ? a : b; }
}