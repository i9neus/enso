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
        UIViewCtx() : resourceMutex(nullptr) {}
        UIViewCtx(std::mutex& mute) : resourceMutex(&mute) {}

        ViewTransform2D         transform;
     
        Cuda::vec2              dragAnchor;
        Cuda::vec2              rotAxis;
        Cuda::vec2              transAnchor;
        float                   rotAnchor;
        float                   scaleAnchor;

        Cuda::vec2              mousePos;
        float                   zoomSpeed;

        std::mutex*             resourceMutex;
    };
}