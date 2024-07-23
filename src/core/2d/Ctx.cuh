#pragma once

#include "core/math/Sampler.cuh"
#include "core/Hash.h"
#include "Transform2D.cuh"

namespace Enso
{
    namespace Host 
    { 
        template<typename T> class Vector; 
        class SceneObject;
    }
    template<typename T> class AssetHandle;
    
    struct UIGridCtx
    {
        __host__ __device__ UIGridCtx() : show(false), majorLineSpacing(0.), minorLineSpacing(0.), lineAlpha(0.f) {}

        bool                    show;
        float                   majorLineSpacing;
        float                   minorLineSpacing;
        float                   lineAlpha;
    };
    
    struct UISelectionCtx
    {
        __host__ UISelectionCtx() {}

        BBox2f                  mouseBBox;
        BBox2f                  lassoBBox;
        BBox2f                  selectedBBox;
        bool                    isLassoing = false;
        std::vector<AssetHandle<Host::SceneObject>> selectedObjects;
        uint                    selectedIdx = 0xffffffff;        

        bool                    isDragging = false;
        vec2                    dragAnchor;
    };

    struct UIViewCtx
    {
        __host__ __device__ UIViewCtx() : dPdXY(0.f), rotAnchor(0.f), scaleAnchor(0.f), zoomSpeed(1.f) {  }

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