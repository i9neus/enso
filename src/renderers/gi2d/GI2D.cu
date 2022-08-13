#include "GI2D.cuh"
#include "generic/Math.h"
#include "kernels/CudaVector.cuh"
#include "kernels/gi2d/CudaGI2DOverlay.cuh"
#include "kernels/gi2d/CudaPrimitive2D.cuh"

using namespace Cuda;

// All renderer objects must be declared and stored in the source file to avoid compilation errors between nvcc and Visual Studio
struct CudaObjects
{
    Cuda::AssetHandle<Cuda::Host::GI2DOverlay>                  overlayRenderer;
    Cuda::AssetHandle<Cuda::Host::Vector<Cuda::LineSegment>>    hostLineSegments;

    Cuda::GI2DOverlayParams                                     overlayParams;
    Cuda::AssetHandle<Cuda::Host::BIH2DAsset>                   sceneBIH;
    Cuda::AssetHandle<Cuda::Host::BIH2DAsset>                   newObjctBIH;
};

GI2D::GI2D() :
    m_objectsPtr(std::make_unique<CudaObjects>()),
    m_objects(*m_objectsPtr)
{
    m_uiGraph.DeclareState("kIdleState", this, &GI2D::OnIdleState);

    // Create path
    m_uiGraph.DeclareState("kCreatePathOpen", this, &GI2D::OnCreatePath);      
    m_uiGraph.DeclareState("kCreatePathHover", this, &GI2D::OnCreatePath);
    m_uiGraph.DeclareState("kCreatePathAppend", this, &GI2D::OnCreatePath);
    m_uiGraph.DeclareState("kCreatePathClose", this, &GI2D::OnCreatePath);
    m_uiGraph.DeclareDeterministicTransition("kIdleState", "kCreatePathOpen", KeyboardButtonMap({ {'Q', kOnButtonDepressed}, {VK_CONTROL, kButtonDown} }), nullptr, 0);
    m_uiGraph.DeclareDeterministicTransition("kCreatePathHover", "kCreatePathAppend", nullptr, MouseButtonMap(kMouseLButton, kOnButtonDepressed), 0);
    m_uiGraph.DeclareDeterministicTransition("kCreatePathHover", "kCreatePathClose", KeyboardButtonMap(VK_ESCAPE, kOnButtonDepressed), nullptr, 0);

    // Select path
    m_uiGraph.DeclareState("kSelectPathDragging", this, &GI2D::OnSelectPath);
    m_uiGraph.DeclareState("kSelectPathEnd", this, &GI2D::OnSelectPath);
    m_uiGraph.DeclareDeterministicTransition("kIdleState", "kSelectPathDragging", nullptr, MouseButtonMap(kMouseLButton, kOnButtonDepressed), 0);
    m_uiGraph.DeclareDeterministicTransition("kSelectPathDragging", "kSelectPathDragging", nullptr, MouseButtonMap(kMouseLButton, kButtonDown), kUITriggerOnMouseMove);
    m_uiGraph.DeclareDeterministicTransition("kSelectPathDragging", "kSelectPathEnd", nullptr, MouseButtonMap(kMouseLButton, kOnButtonReleased), 0);

    m_uiGraph.Finalise();
}

GI2D::~GI2D()
{
    Destroy();
}

uint GI2D::OnIdleState(const UIStateTransition& transition)
{
    Log::Success("Back home!");
    return kUIStateOkay;
}

uint GI2D::OnSelectPath(const UIStateTransition& transition)
{   
    if (!transition.HasDeterministicTarget()) { return kUIStateError; }

    const std::string stateID = m_uiGraph.GetTargetStateID(transition);
    if (stateID == "kSelectPathDragging")
    {
        auto& selection = m_objects.overlayParams.selection;
        if (!selection.showBounds)
        {
            selection.bBox = BBox2f(m_objects.overlayParams.mousePosView, m_objects.overlayParams.mousePosView);
            selection.showBounds = true;            
        }
        else
        {
            selection.bBox[1] = m_objects.overlayParams.mousePosView;
        }
        Log::Success("Dragging!");
    }
    else if (stateID == "kSelectPathEnd")
    {
        m_objects.overlayParams.selection.showBounds = false;
        Log::Success("Finished!");

        m_uiGraph.SetState("kIdleState");
    }
    else
    {
        return kUIStateError;        
    }

    SetDirtyFlags(kGI2DDirtyParams);
    return kUIStateOkay;
}

uint GI2D::OnCreatePath(const UIStateTransition& transition)
{
    if (!transition.HasDeterministicTarget()) { return kUIStateError; }

    const std::string stateID = m_uiGraph.GetTargetStateID(transition);
    if (stateID == "kCreatePathOpen")
    {
        // Record the index of the starting segment on the path 
        m_newPath.pathStartIdx = m_objects.hostLineSegments->Size() - 1;
        m_newPath.numVertices = 0;
        
        m_uiGraph.SetState("kCreatePathHover");
    }
    else if (stateID == "kCreatePathHover")
    {
        if (m_newPath.numVertices >= 2)
        {
            m_objects.hostLineSegments->Back()[1] = m_objects.overlayParams.mousePosView;
            SetDirtyFlags(kGI2DDirtyLineSegments);
        }
    }
    else if (stateID == "kCreatePathAppend")
    {
        if (m_newPath.numVertices == 0)
        {
            // Create a zero-length segment that will be manipulated later
            m_objects.hostLineSegments->EmplaceBack(m_objects.overlayParams.mousePosView, m_objects.overlayParams.mousePosView, 0);
            m_newPath.numVertices = 2;
            SetDirtyFlags(kGI2DDirtyLineSegments);
        }
        else if (m_newPath.numVertices >= 2)
        {
            // Any more and we simply reuse the last vertex on the path as the start of the next segment
            m_objects.hostLineSegments->EmplaceBack(m_objects.hostLineSegments->Back()[1], m_objects.overlayParams.mousePosView, 0);
            m_newPath.numVertices++;
            SetDirtyFlags(kGI2DDirtyLineSegments);
        }

        m_uiGraph.SetState("kCreatePathHover");
    }
    else if (stateID == "kCreatePathClose")
    {
        m_uiGraph.SetState("kIdleState");
    }
    else
    {
        return kUIStateError;
    }
    
    return kUIStateOkay;
}

std::shared_ptr<RendererInterface> GI2D::Instantiate()
{
    return std::make_shared<GI2D>();
}

void GI2D::RebuildBIH()
{
    // Synchronise the segments
    Host::Vector<LineSegment>& segments = *m_objects.hostLineSegments;
    segments.Synchronise(kVectorSyncUpload);

    // Create a segment list ready for building
    // TODO: It's probably faster if we build on the already-sorted index list
    auto& primIdxs = m_objects.sceneBIH->GetPrimitiveIndices();
    primIdxs.resize(segments.Size());
    for (uint idx = 0; idx < primIdxs.size(); ++idx) { primIdxs[idx] = idx; }

    // Construct the BIH
    std::function<BBox2f(uint)> getPrimitiveBBox = [&segments](const uint& idx) -> BBox2f
    {
        return Grow(segments[idx].GetBoundingBox(), 0.001f);
    };
    m_objects.sceneBIH->Build(getPrimitiveBBox);

    SetDirtyFlags(kGI2DDirtyLineSegments);
}

void GI2D::OnInitialise()
{
    m_view.trans = vec2(0.f);
    m_view.scale = 1.0f;
    m_view.rotate = 0.0;
    m_objects.overlayParams.viewMatrix = ConstructViewMatrix(m_view.trans, m_view.rotate, m_view.scale) * m_clientToNormMatrix;
    m_view.zoomSpeed = 10.0f;   

    //m_primitiveContainer.Create(m_renderStream);

    m_objects.hostLineSegments = CreateAsset<Host::Vector<LineSegment>>("id_lineSegments", kVectorHostAlloc, m_renderStream);
    
    m_objects.sceneBIH = CreateAsset<Host::BIH2DAsset>("id_gi2DBIH");
    m_objects.overlayRenderer = CreateAsset<Host::GI2DOverlay>("id_gi2DOverlay", m_objects.sceneBIH, m_objects.hostLineSegments);

    SetDirtyFlags(kGI2DDirtyLineSegments | kGI2DDirtyParams);
}

void GI2D::OnDestroy()
{
    m_objects.overlayRenderer.DestroyAsset();
    m_objects.sceneBIH.DestroyAsset();
    m_objects.hostLineSegments.DestroyAsset();
}

void GI2D::OnRender()
{
    //std::this_thread::sleep_for(std::chrono::milliseconds(50));
    //Log::Write("Tick");

    if (m_dirtyFlags & kGI2DDirtyParams)
    {
        std::lock_guard <std::mutex> lock(m_resourceMutex);

        m_objects.overlayRenderer->SetParams(m_objects.overlayParams);

        ClearDirtyFlags(kGI2DDirtyParams);
    }
    if (m_dirtyFlags & kGI2DDirtyLineSegments)
    {
        RebuildBIH();

        ClearDirtyFlags(kGI2DDirtyLineSegments);
    }

    m_objects.overlayRenderer->Render(m_compositeImage);
}

void GI2D::OnKey(const uint code, const bool isSysKey, const bool isDown)
{

}

void GI2D::OnMouseButton(const uint code, const bool isDown)
{
    // Is the view being changed? 
    if (code == kMouseMButton)
    {
        m_view.dragAnchor = vec2(m_mouse.pos);
        m_view.rotAxis = normalize(m_view.dragAnchor - vec2(m_clientWidth, m_clientHeight) * 0.5f);
        m_view.transAnchor = m_view.trans;
        m_view.scaleAnchor = m_view.scale;
        m_view.rotAnchor = m_view.rotate;
    } 
}

mat3 GI2D::ConstructViewMatrix(const vec2& trans, const float rotate, const float scale) const
{
    const float sinTheta = std::sin(rotate);
    const float cosTheta = std::cos(rotate);
    mat3 m = mat3::Indentity();
    m.i00 = scale * cosTheta; m.i01 = scale * sinTheta;
    m.i10 = scale * sinTheta; m.i11 = scale * -cosTheta;
    m.i02 = trans.x;
    m.i12 = trans.y;
    return m;
}

void GI2D::OnMouseMove()
{
    // Dragging?
    if (IsMouseButtonDown(kMouseMButton))
    {
        OnViewChange();
    }
    
    {
        std::lock_guard <std::mutex> lock(m_resourceMutex);
        m_objects.overlayParams.mousePosView = m_objects.overlayParams.viewMatrix * vec2(m_mouse.pos);
    }

    // Mark the scene as dirty
    SetDirtyFlags(kGI2DDirtyParams);
}

void GI2D::OnViewChange()
{
    // Zooming?
    if (IsKeyDown(VK_CONTROL))
    {
        float logScaleAnchor = std::log2(::math::max(1e-10f, m_view.scaleAnchor));
        logScaleAnchor += m_view.zoomSpeed * float(m_mouse.pos.y - m_view.dragAnchor.y) / m_clientHeight;
        m_view.scale = std::pow(2.0, logScaleAnchor);

        //Log::Write("Scale: %f", m_view.scale);
    }
    // Rotating?
    else if (IsKeyDown(VK_SHIFT))
    {
        const vec2 delta = normalize(vec2(m_mouse.pos) - vec2(m_clientWidth, m_clientHeight) * 0.5f);
        const float theta = std::acos(dot(delta, m_view.rotAxis)) * (float(dot(delta, vec2(m_view.rotAxis.y, -m_view.rotAxis.x)) < 0.0f) * 2.0 - 1.0f);
        m_view.rotate = m_view.rotAnchor + theta;

        if (std::abs(std::fmod(m_view.rotate, kHalfPi)) < 0.05f) { m_view.rotate = std::round(m_view.rotate / kHalfPi) * kHalfPi; }

        //Log::Write("Theta: %f", m_view.rotate);
    }
    // Translating
    else
    {
        // Update the transformation
        m_objects.overlayParams.viewMatrix = ConstructViewMatrix(m_view.transAnchor, m_view.rotate, m_view.scale) * m_clientToNormMatrix;
        const vec2 dragDelta = (m_objects.overlayParams.viewMatrix * vec2(m_view.dragAnchor)) - (m_objects.overlayParams.viewMatrix * vec2(m_mouse.pos));
        m_view.trans = m_view.transAnchor + dragDelta;

        //Log::Write("Trans: %s", m_view.trans.format());
    }

    // Update the parameters in the overlay renderer
    {
        std::lock_guard <std::mutex> lock(m_resourceMutex);
        m_objects.overlayParams.viewMatrix = ConstructViewMatrix(m_view.trans, m_view.rotate, m_view.scale) * m_clientToNormMatrix;
        m_objects.overlayParams.viewScale = m_view.scale;
    }
}

void GI2D::OnMouseWheel()
{

}

void GI2D::OnResizeClient()
{
}