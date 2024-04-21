#pragma once

#include "core/math/Sampler.cuh"
#include "core/Hash.h"
#include "Transform2D.cuh"
#include "FwdDecl.cuh"

namespace Enso
{
    namespace Host { template<typename T> class Vector; }
    
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
     
        vec2                    dragAnchor;
        vec2                    rotAxis;
        vec2                    transAnchor;
        float                   rotAnchor;
        float                   scaleAnchor;

        vec2                    mousePos;
        float                   zoomSpeed;
    };
}