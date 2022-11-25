#include "GI2DModule.h"
#include "thirdparty/imgui/backends/imgui_impl_dx12.h"
#include "thirdparty/imgui/backends/imgui_impl_win32.h"

namespace Enso
{
    /*void GI2DUI::ConstructToolbox()
    {

    }*/
    
    void GI2DUI::ConstructComponent()
    {
        ImGui::Begin(m_componentId.c_str());
        m_commandQueue.BeginComponent(m_componentId);
        
        if (ImGui::Button("Draw"))
        {
            m_commandQueue.BeginCommand("draw");
        }

        ImGui::End();
    }
}