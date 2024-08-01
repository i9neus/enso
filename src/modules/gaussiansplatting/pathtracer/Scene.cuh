#pragma once

#include "core/math/Math.cuh"
#include "core/math/Hash.cuh"

namespace Enso
{
    #define kEmitterPos vec3(0., 0.5, 0.5)
    #define kEmitterRot vec3(kHalfPi * 1.5, 0., 0.)
    #define kEmitterSca 1.
    #define kEmitterPower 2.
    #define kEmitterRadiance (kOne * kEmitterPower / sqr(kEmitterSca))
}