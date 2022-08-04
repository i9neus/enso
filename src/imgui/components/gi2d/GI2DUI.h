#pragma once

#include "../ComponentInterface.h"

namespace Gui
{
    class GI2DUI : public ComponentInterface
    {
    public:
        GI2DUI(Json::CommandQueue& commandQueue) : ComponentInterface("gi2d", commandQueue) {}

        virtual void ConstructComponent() override final;
    };
}