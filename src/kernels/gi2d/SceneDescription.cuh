#pragma once

#include "FwdDecl.cuh"
#include "tracables/Tracable.cuh"

namespace GI2D
{
    namespace Device
    {
        struct SceneDescription
        {
            Core::Vector<Device::Tracable*>*    tracables = nullptr;
            BIH2D<BIH2DFullNode>*               bih = nullptr;
        };
    }
}