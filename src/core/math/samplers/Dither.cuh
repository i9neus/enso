#pragma once

#include "../Math.cuh"

namespace Enso
{
    /*__host__ __device__ float OrderedDither(ivec2 xyScreen)
    {
        // NOTE: nvcc *should* optimise this down into a indexed branch table
        switch (xyScreen.x & 3)
        {
        case 0:
            switch (xyScreen.y & 3)
            {
            case 0: return 0.;
            case 1: return (8. + 0.5) / 16.;
            case 2: return (2. + 0.5) / 16.;
            case 3: return (10. + 0.5) / 16.;
            }
        case 1:
            switch (xyScreen.y & 3)
            {
            case 0: return (12. + 0.5) / 16.;
            case 1: return (4. + 0.5) / 16.;
            case 2: return (14. + 0.5) / 16.;
            case 3: return (6. + 0.5) / 16.;
            }
        case 2:
            switch (xyScreen.y & 3)
            {
            case 0: return (3. + 0.5) / 16.;
            case 1: return (11. + 0.5) / 16.;
            case 2: return (1. + 0.5) / 16.;
            case 3: return (9. + 0.5) / 16.;
            }
        case 3:
            switch (xyScreen.y & 3)
            {
            case 0: return (15. + 0.5) / 16.;
            case 1: return (7. + 0.5) / 16.;
            case 2: return (13. + 0.5) / 16.;
            case 3: return (5. + 0.5) / 16.;
            }
        }
        return 0.;
    }*/
    
    __host__ __device__ float OrderedDither(const ivec2& xyScreen)
    {
        // NOTE: Not clear whether this is faster or slower than using a branch table
        const mat4 kOrderedDither = mat4(vec4(0.0, 8.0, 2.0, 10.), vec4(12., 4., 14., 6.), vec4(3., 11., 1., 9.), vec4(15., 7., 13., 5.));
        return (kOrderedDither[xyScreen.x & 3][xyScreen.y & 3] + 0.5) / 16.0;
    }
}