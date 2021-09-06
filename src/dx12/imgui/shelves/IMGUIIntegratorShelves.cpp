#include "IMGUIIntegratorShelves.h"
#include <random>

WavefrontTracerShelf::WavefrontTracerShelf(const Json::Node& json) :
    IMGUIShelf(json)
{}

void WavefrontTracerShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ImGui::DragFloat("Russian roulette", &m_p.russianRouletteThreshold, 0.001f, 0.0f, 1.0f, "%.4f");
    HelpMarker("The Russian roulette threshold specifies the minimum throughput weight below which rays are probabilsitically terminated.");

    ConstructComboBox("Direct sampling", { "MIS", "Lights", "BxDFs" }, m_p.importanceMode);
    HelpMarker("Specifies the sampling method for direct lights.");

    //ConstructComboBox("Trace mode", { "Wavefront", "Path" }, m_p.traceMode);
    //HelpMarker("Speciffies whether to use wavefront tracing or interative path tracing.");

    ConstructComboBox("Light selection mode", { "Naive", "Weighted" }, m_p.lightSelectionMode);
    HelpMarker("Specifies how direct lights should be selected. Naive mode selects lights at random. Weighted mode picks lights based on their estimated contribution.");

    ConstructComboBox("Shading mode", { "Default", "Simple", "Normals", "Debug" }, m_p.shadingMode);
    HelpMarker("Specifies how objects are shaded. Default performs full physically based shading. Simple reverts to a basic lighting model. Normals outputs the surface normals only. Debug outputs shader diagnostics.");
}

void WavefrontTracerShelf::Jitter(const uint flags, const uint operation)
{
    if (!(flags & kStatePermuteColours)) { return; }
}