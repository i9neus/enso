#include "IMGUILightShelves.h"
#include <random>

QuadLightShelf::QuadLightShelf(const Json::Node& json) :
    IMGUIShelf(json),
    m_colourPicker(m_p.light.colourHSV, "Colour"),
    m_intensity(m_p.light.intensity, "Intensity", "The intensity of the light in stops.", Cuda::vec3(-10.0f, 10.0f, 1.0f)),
    m_flags(m_p.light.renderObject.flags, "Light flags")
{
    m_flags.Initialise({ "Visible", "Exclude from bake" });
}

void QuadLightShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    m_flags.Construct();
    ConstructJitteredTransform(m_p.light.transform, true);
    m_colourPicker.Construct();
    m_intensity.Construct();
}

void QuadLightShelf::Jitter(const uint flags, const uint operation)
{
    if (!(flags & kStatePermuteLights)) { return; }

    if (flags & kStatePermuteColours)
    {
        m_p.light.colourHSV.Update(operation);
        m_p.light.intensity.Update(operation);
    }
    if (flags & kStatePermuteTransforms) { m_p.light.transform.Update(operation); }
    if (flags & kStatePermuteObjectFlags) { m_p.light.renderObject.flags.Update(operation); }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

SphereLightShelf::SphereLightShelf(const Json::Node& json) :
    IMGUIShelf(json),
    m_colourPicker(m_p.light.colourHSV, "Colour"),
    m_intensity(m_p.light.intensity, "Intensity", "The intensity of the light in stops.", Cuda::vec3(-10.0f, 10.0f, 1.0f)),
    m_flags(m_p.light.renderObject.flags, { "Light flags" })
{
    m_flags.Initialise({ "Visible", "Exclude from bake" });
}

void SphereLightShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    m_flags.Construct();
    ConstructJitteredTransform(m_p.light.transform, true);
    m_colourPicker.Construct();
    m_intensity.Construct();
}

void SphereLightShelf::Jitter(const uint flags, const uint operation)
{
    if (!(flags & kStatePermuteLights)) { return; }

    if (flags & kStatePermuteColours)
    {
        m_p.light.colourHSV.Update(operation);
        m_p.light.intensity.Update(operation);
    }
    if (flags & kStatePermuteTransforms) { m_p.light.transform.Update(operation); }
    if (flags & kStatePermuteObjectFlags) { m_p.light.renderObject.flags.Update(operation); }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

DistantLightShelf::DistantLightShelf(const Json::Node& json) :
    IMGUIShelf(json),
    m_colourPicker(m_p.light.colourHSV, "Colour"),
    m_intensity(m_p.light.intensity, "Intensity", "The intensity of the light in stops.", Cuda::vec3(-10.0f, 10.0f, 1.0f)),
    m_flags(m_p.light.renderObject.flags, { "Light flags" })
{
    m_flags.Initialise({ "Visible", "Exclude from bake" });
}

void DistantLightShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    m_flags.Construct();
    ConstructJitteredTransform(m_p.light.transform, true);

    m_colourPicker.Construct();
    m_intensity.Construct();

    ImGui::DragFloat("Angle", &m_p.angle, math::max(0.01f, m_p.angle * 0.01f), 0.0f, std::numeric_limits<float>::max());
    HelpMarker("The projected solid angle of the light in degrees.");
}

void DistantLightShelf::Jitter(const uint flags, const uint operation)
{
    if (!(flags & kStatePermuteLights)) { return; }

    if (flags & kStatePermuteColours)
    {
        m_p.light.colourHSV.Update(operation);
        m_p.light.intensity.Update(operation);
    }
    if (flags & kStatePermuteTransforms) { m_p.light.transform.Update(operation); }
    if (flags & kStatePermuteObjectFlags) { m_p.light.renderObject.flags.Update(operation); }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

EnvironmentLightShelf::EnvironmentLightShelf(const Json::Node& json) :
    IMGUIShelf(json),
    m_colourPicker(m_p.light.colourHSV, "Colour"),
    m_intensity(m_p.light.intensity, "Intensity", "The intensity of the light in stops.", Cuda::vec3(-10.0f, 10.0f, 1.0f)),
    m_flags(m_p.light.renderObject.flags, { "Light flags" })
{
    m_flags.Initialise({ "Visible", "Exclude from bake" });
}

void EnvironmentLightShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    m_flags.Construct();
    ConstructJitteredTransform(m_p.light.transform, true);
    m_colourPicker.Construct();
    m_intensity.Construct();
}

void EnvironmentLightShelf::Jitter(const uint flags, const uint operation)
{
    if (!(flags & kStatePermuteLights)) { return; }

    if (flags & kStatePermuteColours)
    {
        m_p.light.colourHSV.Update(operation);
        m_p.light.intensity.Update(operation);
    }
    if (flags & kStatePermuteTransforms) { m_p.light.transform.Update(operation); }
    if (flags & kStatePermuteObjectFlags) { m_p.light.renderObject.flags.Update(operation); }
}