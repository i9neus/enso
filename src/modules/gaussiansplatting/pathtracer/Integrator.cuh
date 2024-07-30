#pragma once

#include "core/3d/Ctx.cuh"

namespace Enso
{
    enum MaterialType : int
    {
        kMatInvalid = -1,
        kMatEmitter = 0,
        kMatRoughSpecular = 1,
        kMatRoughDielectric = 2,
        kMatPerfectSpecular = 3,
        kMatPerfectDielectric = 4,
        kMatLambertian = 5,
        kMatCompound = 6
    };
}