#pragma once

#include "../CudaSampler.cuh"
#include "generic/Hash.h"
#include "Transform2D.cuh"

using namespace Cuda;

namespace Cuda
{
	namespace Host { template<typename T> class Vector; }
}

namespace GI2D
{
    struct UIGridCtx
    {
        bool                    show;
        float                   majorLineSpacing;
        float                   minorLineSpacing;
        float                   lineAlpha;
    };
    
    struct UISelectionCtx
    {
        BBox2f                  mouseBBox;
        BBox2f                  lassoBBox;
        BBox2f                  selectedBBox;
        bool                    isLassoing = false;
        uint                    numSelected = 0;
        uint                    selectedIdx = 0xffffffff;
    };

    struct UIViewCtx
    {
        __host__ __device__ UIViewCtx() {  }

        __host__ __device__ void Prepare()
        {
            dPdXY = length(vec2(transform.matrix.i00, transform.matrix.i10));
        }

        ViewTransform2D         transform;

        BBox2f                  sceneBounds;
        float                   dPdXY;
     
        Cuda::vec2              dragAnchor;
        Cuda::vec2              rotAxis;
        Cuda::vec2              transAnchor;
        float                   rotAnchor;
        float                   scaleAnchor;

        Cuda::vec2              mousePos;
        float                   zoomSpeed;
    };
}