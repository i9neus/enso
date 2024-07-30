#pragma once

#include "core/math/Math.cuh"

namespace Enso
{
    // Origin/direction 
    struct RayBasic
    {
        vec3            o;
        vec3            d;
    };

    // Origin/direction plus attributes
    struct Ray
    {
        RayBasic        od;
        float           tNear;
        vec3            weight;
        float           pdf;
    };
}