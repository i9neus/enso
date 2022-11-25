#pragma once

#include "../UIModuleInterface.h"

namespace Enso
{
    class GI2DUI : public UIModuleInterface
    {
    public:
        GI2DUI(Json::CommandQueue& commandQueue) : UIModuleInterface("gi2d", commandQueue) {}

        virtual void ConstructComponent() override final;
    };
}