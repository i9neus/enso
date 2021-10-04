#include "IMGUITracableShelves.h"
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

///////////////////////////////////////////////////////////////////////////////////////////////////

SDFShelf::SDFShelf(const Json::Node& json) :
    IMGUIShelf(json),
    m_jitteredParamTable("KIFS Params")
{
   
}

void SDFShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ConstructJitteredTransform(m_p.transform, true);

    m_jitteredParamTable.Construct();

    ConstructComboBox("SDF Primitive", std::vector<std::string>({ "Sphere", "Torus", "Box" }), m_p.primitiveType);
    HelpMarker("The shape of the SDF clip object.");

    if (m_p.primitiveType == Cuda::kSDFPrimitiveSphere)
    {
        ImGui::DragFloat("Radius", &m_p.sphere.r, math::max(0.001f, m_p.sphere.r * 0.01f), 0.0f, std::numeric_limits<float>::max(), "%.4f");
    }
    else if (m_p.primitiveType == Cuda::kSDFPrimitiveTorus)
    {
        ImGui::DragFloat("Radius 1", &m_p.torus.r1, math::max(0.001f, m_p.torus.r1 * 0.01f), 0.0f, std::numeric_limits<float>::max(), "%.4f");
        ImGui::DragFloat("Radius 2", &m_p.torus.r2, math::max(0.001f, m_p.torus.r2 * 0.01f), 0.0f, std::numeric_limits<float>::max(), "%.4f");
    }
    else if (m_p.primitiveType == Cuda::kSDFPrimitiveBox)
    {
        ImGui::DragFloat("Size", &m_p.box.size, math::max(0.001f, m_p.box.size * 0.01f), 0.0f, std::numeric_limits<float>::max(), "%.4f");
    }

    ImGui::DragInt("Max Specular Interations", &m_p.maxSpecularIterations, 1, 1, 500);
    HelpMarker("The maximum number of iterations applied to specular rays");

    ImGui::DragInt("Max Diffuse Iterations", &m_p.maxDiffuseIterations, 1, 1, 500);
    HelpMarker("The maximum number of iterations applied to diffuse rays.");

    ImGui::DragFloat("Cutoff Threshold", &m_p.cutoffThreshold, math::max(0.00001f, m_p.cutoffThreshold * 0.01f), 0.0f, 1.0f, "%.6f");
    HelpMarker("The threshold below which the ray is deemed to have interected the isosurface.");

    ImGui::DragFloat("Escape Threshold", &m_p.escapeThreshold, math::max(0.01f, m_p.escapeThreshold * 0.01f), 0.0f, 5.0f);
    HelpMarker("The threshold at which a sample is deemed to have left the SDF's field of influence.");

    ImGui::DragFloat("Ray Increment", &m_p.rayIncrement, math::max(0.01f, m_p.rayIncrement * 0.01f), 0.0f, 2.0f);
    HelpMarker("The increment multiplier applied at the marching step. Ideally, this value should be set to 1.");

    ImGui::DragFloat("Ray Kickoff", &m_p.rayKickoff, math::max(0.01f, m_p.rayKickoff * 0.01f), 0.0f, 1.0f);
    HelpMarker("The kickoff applied to child rays that are spawned from the surface of the SDF.");

    ImGui::DragFloat("Fail Threshold", &m_p.failThreshold, math::max(0.00001f, m_p.failThreshold * 0.01f), 0.0f, 1.0f, "%.6f");
    HelpMarker("The threshold at which the value of the SDF invalidates the intersection. Raise this value to remove visible holes in the field.");
}

void SDFShelf::Update()
{

}

void SDFShelf::Jitter(const uint flags, const uint operation)
{
    if (!(flags & kStatePermuteGeometry)) { return; }

    if (flags & kStatePermuteTransforms) { m_p.transform.Update(operation); }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

PlaneShelf::PlaneShelf(const Json::Node& json) :
    IMGUIShelf(json),
    m_flags(m_p.tracable.renderObject.flags, "Object flags")
{
    m_flags.Initialise(std::vector<std::string>({ "Visible", "Exclude from bake" }));
}

void PlaneShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    m_flags.Construct();
    ConstructJitteredTransform(m_p.tracable.transform, true);
    
    ImGui::Checkbox("Bounded", &m_p.isBounded);
    HelpMarker("Check to bound the plane in the range [-0.5, 0.5] in object space. Uncheck to trace to infinity in all directions.");
}

void PlaneShelf::Jitter(const uint flags, const uint operation)
{
    if (!(flags & kStatePermuteGeometry)) { return; }

    if (flags & kStatePermuteObjectFlags) { m_p.tracable.renderObject.flags.Update(operation); }
    if (flags & kStatePermuteTransforms) { m_p.tracable.transform.Update(operation); }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

SphereShelf::SphereShelf(const Json::Node& json) :
    IMGUIShelf(json),
    m_flags(m_p.renderObject.flags, "Object flags")
{
    m_flags.Initialise(std::vector<std::string>({ "Visible", "Exclude from bake" }));
}

void SphereShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    m_flags.Construct();
    ConstructJitteredTransform(m_p.transform, true);
}

void SphereShelf::Jitter(const uint flags, const uint operation)
{
    if (!(flags & kStatePermuteGeometry)) { return; }

    if (flags & kStatePermuteObjectFlags) { m_p.renderObject.flags.Update(operation); }
    if (flags & kStatePermuteTransforms) { m_p.transform.Update(operation); }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

BoxShelf::BoxShelf(const Json::Node& json) :
    IMGUIShelf(json),
    m_flags(m_p.renderObject.flags, "Object flags")
{
    m_flags.Initialise(std::vector<std::string>({ "Visible", "Exclude from bake" }));
}

void BoxShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    m_flags.Construct();
    ConstructJitteredTransform(m_p.transform, true, true);
}

void BoxShelf::Jitter(const uint flags, const uint operation)
{
    if (!(flags & kStatePermuteGeometry)) { return; }

    if (flags & kStatePermuteObjectFlags) { m_p.renderObject.flags.Update(operation); }
    if (flags & kStatePermuteTransforms) { m_p.transform.Update(operation); }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

CornellBoxShelf::CornellBoxShelf(const Json::Node& json) :
    IMGUIShelf(json),
    m_flags(m_p.tracable.renderObject.flags, "Object flags"),
    m_faceMask(m_p.faceMask, "Face mask"),
    m_cameraRayMask(m_p.cameraRayMask, "Camera ray mask")
{
    m_flags.Initialise(std::vector<std::string>({ "Visible", "Exclude from bake" }));
    m_faceMask.Initialise(std::vector<std::string>({ "1", "2", "3", "4", "5", "6" }));
    m_cameraRayMask.Initialise(std::vector<std::string>({ "1", "2", "3", "4", "5", "6" }));
}

void CornellBoxShelf::Construct()
{
    if (!ImGui::CollapsingHeader(GetShelfTitle().c_str(), ImGuiTreeNodeFlags_DefaultOpen)) { return; }

    ConstructJitteredTransform(m_p.tracable.transform, true);

    m_flags.Construct();
    m_faceMask.Construct();
    m_cameraRayMask.Construct();
}

void CornellBoxShelf::Jitter(const uint flags, const uint operation)
{
    if (!(flags & kStatePermuteGeometry)) { return; }

    if (flags & kStatePermuteObjectFlags) { m_p.tracable.renderObject.flags.Update(operation); }
    if (flags & kStatePermuteTransforms) { m_p.tracable.transform.Update(operation); }

    m_p.faceMask.Update(operation);
    m_p.cameraRayMask.Update(operation);
}
