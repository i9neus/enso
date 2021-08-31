#include "IMGUIKIFSShelf.h"
#include "generic/FilesystemUtils.h"

#include <random>

KIFSShelf::KIFSShelf(const Json::Node& json) : 
    IMGUIShelf(json),
    m_faceFlags(m_p.faceMask, "Faces"),
    m_jitteredParamTable("KIFS Params")
{
    m_faceFlags.Initialise(std::vector<std::string>({ "1", "2", "3", "4", "5", "6" }));
    m_jitteredParamTable.Push("Rotation A", "The degree of rotation applied at each iteration of the fractal.", m_p.rotateA, Cuda::vec3(0.0f, 1.0f, 0.01f));
    m_jitteredParamTable.Push("Rotation B", "The degree of rotation applied at each iteration of the fractal.", m_p.rotateB, Cuda::vec3(0.0f, 1.0f, 0.01f));
    m_jitteredParamTable.Push("Scale A", "The degree of scaling applied at each iteration of the fractal.", m_p.scaleA, Cuda::vec3(0.0f, 1.0f, 0.01f));
    m_jitteredParamTable.Push("Scale B", "The degree of scaling applied at each iteration of the fractal.", m_p.scaleB, Cuda::vec3(0.0f, 1.0f, 0.01f));
    m_jitteredParamTable.Push("Crust thickness", "The thickness of the SDF isosurface.", m_p.crustThickness, Cuda::vec3(0.0f, 1.0f, 0.01f));
    m_jitteredParamTable.Push("Vertex scale", "The relative scale of the SDF primitive.", m_p.vertScale, Cuda::vec3(0.0f, 1.0f, 0.01f));
}

void KIFSShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ConstructJitteredTransform(m_p.transform, true);

    m_jitteredParamTable.Construct();

    ImGui::SliderInt("Iterations ", &m_p.numIterations, 0, kSDFMaxIterations);
    HelpMarker("The number of transformation iterations to apply before evaluating the SDF");

    ConstructComboBox("Fold type", std::vector<std::string>({ "Tetrahedron", "Cube" }), m_p.foldType);
    HelpMarker("The conformal transformation applied to the local space around the SDF.");

    ConstructComboBox("Primitive type", std::vector<std::string>({ "Tetrahedron solid", "Cube solid", "Sphere", "Torus", "Box", "Tetrahedron cage", "Cube cage" }), m_p.primitiveType);
    HelpMarker("The type of SDF primitive that's evaluated after the transformation iterations have been applied. ");

    auto ConstructMaskCheckboxes = [](const std::string& label, uint& value, const int row) -> void
    {
        ImGui::PushID(row);
        for (int i = 0; i < 6; i++)
        {
            bool faceMaskBool = value & (1 << i);
            ImGui::Checkbox(tfm::format("%i", i).c_str(), &faceMaskBool); SL;
            value = (value & ~(1 << i)) | (int(faceMaskBool) << i);
        }
        ImGui::PopID();
        ImGui::Text(label.c_str());
    };

    m_faceFlags.Construct();
    HelpMarker("Enables or disables the faces of SDF primatives which support it.");

    ImGui::Checkbox("SDF Clip Camera Rays", &m_p.sdf.clipCameraRays);
    HelpMarker("Only renders fractal geometry within a bounded space. This helps visaulise the geometric structure in the neighbourhood of the probe volume grid.");

    ConstructComboBox("SDF Clip Shape", std::vector<std::string>({ "Cube", "Sphere", "Torus" }), m_p.sdf.clipShape);
    HelpMarker("The shape of the SDF clip object.");

    ImGui::DragInt("SDF Max Specular Interations", &m_p.sdf.maxSpecularIterations, 1, 1, 500);
    HelpMarker("The maximum number of iterations applied to specular rays");

    ImGui::DragInt("SDF Max Diffuse Iterations", &m_p.sdf.maxDiffuseIterations, 1, 1, 500);
    HelpMarker("The maximum number of iterations applied to diffuse rays.");

    ImGui::DragFloat("SDF Cutoff Threshold", &m_p.sdf.cutoffThreshold, math::max(0.00001f, m_p.sdf.cutoffThreshold * 0.01f), 0.0f, 1.0f, "%.6f");
    HelpMarker("The threshold below which the ray is deemed to have interected the SDF isosurface.");

    ImGui::DragFloat("SDF Escape Threshold", &m_p.sdf.escapeThreshold, math::max(0.01f, m_p.sdf.escapeThreshold * 0.01f), 0.0f, 5.0f);
    HelpMarker("The threshold at which a sample is deemed to have left the SDF's field of influence.");

    ImGui::DragFloat("SDF Ray Increment", &m_p.sdf.rayIncrement, math::max(0.01f, m_p.sdf.rayIncrement * 0.01f), 0.0f, 2.0f);
    HelpMarker("The increment multiplier applied at the marching step. Ideally, this value should be set to 1.");

    ImGui::DragFloat("SDF Ray Kickoff", &m_p.sdf.rayKickoff, math::max(0.01f, m_p.sdf.rayKickoff * 0.01f), 0.0f, 1.0f);
    HelpMarker("The kickoff applied to child rays that are spawned from the surface of the SDF.");

    ImGui::DragFloat("SDF Fail Threshold", &m_p.sdf.failThreshold, math::max(0.00001f, m_p.sdf.failThreshold * 0.01f), 0.0f, 1.0f, "%.6f");
    HelpMarker("The threshold at which the value of the SDF invalidates the intersection. Raise this value to remove visible holes in the field.");
}

void KIFSShelf::Update()
{
    m_faceFlags.Update();
}

void KIFSShelf::Jitter(const uint flags, const uint operation)
{
    if (!(flags & kStatePermuteGeometry)) { return; }
    
    if (flags & kStatePermuteFractals)
    {
        m_p.rotateA.Update(operation);
        m_p.rotateB.Update(operation);
        m_p.scaleA.Update(operation);
        m_p.scaleB.Update(operation);
        m_p.vertScale.Update(operation);
        m_p.crustThickness.Update(operation);
        m_p.faceMask.Update(operation);
    }

    if (flags & kStatePermuteTransforms) { m_p.transform.Update(operation); }
}

void KIFSShelf::JitterKIFSParameters()
{
    
}
