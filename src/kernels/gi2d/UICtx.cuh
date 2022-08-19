#pragma once

#include "../CudaSampler.cuh"
#include "generic/Hash.h"

using namespace Cuda;

namespace Cuda
{
	namespace Host { template<typename T> class Vector; }
}

namespace GI2D
{
    struct UISelectionCtx
    {
        BBox2f                  mouseBBox;
        BBox2f                  lassoBBox;
        BBox2f                  selectedBBox;
        bool                    isLassoing;
        uint                    numSelected;
    };

    struct UIViewCtx
    {
        Cuda::vec2              trans;
        float                   scale;
        float                   rotate;
        float                   zoomSpeed;

        Cuda::vec2              dragAnchor;
        Cuda::vec2              rotAxis;
        Cuda::vec2              transAnchor;
        float                   rotAnchor;
        float                   scaleAnchor;

        Cuda::vec2              mousePos;
        Cuda::mat3              matrix;
        float                   dPdXY;
    };
}