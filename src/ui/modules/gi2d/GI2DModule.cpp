#include "GI2DModule.h"
#include "thirdparty/imgui/backends/imgui_impl_dx12.h"
#include "thirdparty/imgui/backends/imgui_impl_win32.h"

namespace Enso
{
    /*void GI2DUI::ConstructToolbox()
    {

    }*/

    GI2DUI::GI2DUI() : 
        UIModuleInterface("gi2d"),
        m_commandManger(m_objectContainer)
    {}
    
    void GI2DUI::ConstructComponent()
    {
        ImGui::Begin(m_componentId.c_str()); 

        // Flush the command queue and update the object
        if (m_inboundCmdQueue)
        {
            m_commandManger.Flush(*m_inboundCmdQueue);
        }
        
        // Construct the objects
        for (auto& object : m_objectContainer)
        {
            object.second->Construct();
        }

        ImGui::End();
    }
}