#pragma once

#include <stdint.h>
#include <limits>
#include <math.h>
#include "Constants.h"

#undef min
#undef max

namespace math
{
    template<typename T> inline constexpr T mod2(const T& a, const T& b)    { return ((a % b) + b) % b; }
    template<typename T> inline constexpr T sqr(const T& x)                 { return x * x; }
    template<typename T> inline constexpr T cub(const T& x)                 { return x * x * x; }
    template<typename T> inline T min(const T& a, const T& b)               { return (a < b) ? a : b; }
    template<typename T> inline T max(const T& a, const T& b)               { return (a > b) ? a : b; }
    template<typename T> inline T clamp(const T& v, const T& a, const T& b) { return math::max(a, math::min(b, v));  }
}