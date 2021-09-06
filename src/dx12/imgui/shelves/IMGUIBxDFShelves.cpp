#include "IMGUIBxDFShelves.h"
#include <random>

void LambertBRDFShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str())) { return; }

    ImGui::PushItemWidth(50);
    
    ImGui::SliderInt("Probe volume grid", &m_p.lightProbeGridIdx, 0, 1);
    HelpMarker("Selects the evaulated light probe grid depending on the bake setting. 0 = direct or combined, 1 = indirect");

    ImGui::PopItemWidth();
}
