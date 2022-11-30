#include "GI2DModule.h"
#include "thirdparty/imgui/backends/imgui_impl_dx12.h"
#include "thirdparty/imgui/backends/imgui_impl_win32.h"

namespace Enso
{
    /*void GI2DUI::ConstructToolbox()
    {

    }*/

    GI2DUI::GI2DUI(std::shared_ptr<CommandQueue> outQueue) :
        UIModuleInterface("gi2d", outQueue),
        m_commandManger(m_objectContainer)
    {
        Assert(outQueue);

        outQueue->RegisterCommand("OnUpdateObject");
    }
    
    void GI2DUI::ConstructComponent()
    {
        ImGui::Begin(m_componentId.c_str()); 

        // Flush the command queue and update the object
        if (m_inboundCmdQueue)
        {
            m_commandManger.Flush(*m_inboundCmdQueue);
        }
        
        // Construct the objects
        std::vector<UIGenericObject*> dirtyObjects;
        for (auto& object : m_objectContainer)
        {
            if (object.second->Construct())
            {
                dirtyObjects.push_back(object.second.get());
            }
        }
        
        // If any are dirty, update them now
        if (!dirtyObjects.empty())
        {
            m_outboundCmdQueue->Clear();
            Json::Node cmdNode = m_outboundCmdQueue->Create("OnUpdateObject");
            for (auto& object : dirtyObjects)
            {
                object->Serialise(cmdNode);
            }
            m_outboundCmdQueue->Enqueue();
            Log::Success(m_outboundCmdQueue->Format());
            Log::Write("Finished!");
        }

        ImGui::End();
    }
}