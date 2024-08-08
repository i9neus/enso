#pragma once

#include "core/CudaHeaders.cuh"

namespace Enso
{   
    namespace Host
    {
        // Renderable objects are designed to be rapidly cycled by the inner loop
        class RenderableObject
        {
        public:
            __host__ RenderableObject() = default;

            virtual void Render() = 0;
        };
    }
}
