#include "IMGUIBxDFShelves.h"
#include <random>

void LambertBRDFShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str())) { return; }

    ImGui::PushItemWidth(50);

    ConstructFlagCheckBox("Use light probe grid", Cuda::kLambertUseProbeGrid, m_p.probeGridFlags);
    if (m_p.probeGridFlags & Cuda::kLambertUseProbeGrid)
    {
        ConstructComboBox("Max SH order", { "0", "1", "2" }, m_p.maxSHOrder);
        HelpMarker("Specifies the maximum spherical harmonic order to use.");
        
        ImGui::Text("Channels"); SL;
        for (int i = 0; i < Cuda::kLambertGridNumChannels; ++i)
        {
            ConstructFlagCheckBox(tfm::format("%i", i), 1u << i, m_p.probeGridFlags);
            if (i < Cuda::kLambertGridNumChannels - 1) { SL; }
        }
    }
    //HelpMarker("Selects the evaulated light probe grid depending on the bake setting. 0 = direct or combined, 1 = indirect");

    ImGui::PopItemWidth();
}
