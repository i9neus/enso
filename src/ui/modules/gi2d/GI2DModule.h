#pragma once

#include "../UIModuleInterface.h"
#include "ui/elements/UIGenericObject.h"
#include "ui/elements/UICommandManager.h"

namespace Enso
{
    class GI2DUI : public UIModuleInterface
    {
    public:
        GI2DUI();

        virtual void ConstructComponent() override final;

    private:
        UIObjectContainer   m_objectContainer;
        UICommandManager    m_commandManger;        
    };
}