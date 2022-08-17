#include "GI2DRenderer.cuh"
#include "generic/Math.h"
#include "kernels/CudaVector.cuh"
#include "kernels/gi2d/CudaGI2DOverlay.cuh"
#include "kernels/gi2d/CudaGI2DPathTracer.cuh"

using namespace Cuda;
using namespace GI2D;

// All renderer objects must be declared and stored in the source file to avoid compilation errors between nvcc and Visual Studio
struct CudaObjects
{
    AssetHandle<GI2D::Host::Overlay>                overlayRenderer;
    AssetHandle<GI2D::Host::PathTracer>             pathTracer;
    AssetHandle<Cuda::Host::Vector<GI2D::LineSegment>> hostLineSegments;

    GI2D::OverlayParams                             overlayParams;
    GI2D::PathTracerParams                          pathTracerParams;
    AssetHandle<GI2D::Host::BIH2DAsset>             sceneBIH;
    AssetHandle<GI2D::Host::BIH2DAsset>             newObjctBIH;

    GI2D::ViewTransform                             viewTransform;
};

GI2DRenderer::GI2DRenderer() :
    m_objectsPtr(std::make_unique<CudaObjects>()),
    m_objects(*m_objectsPtr)
{
    m_uiGraph.DeclareState("kIdleState", this, &GI2DRenderer::OnIdleState);

    // Create path
    m_uiGraph.DeclareState("kCreatePathOpen", this, &GI2DRenderer::OnCreatePath);      
    m_uiGraph.DeclareState("kCreatePathHover", this, &GI2DRenderer::OnCreatePath);
    m_uiGraph.DeclareState("kCreatePathAppend", this, &GI2DRenderer::OnCreatePath);
    m_uiGraph.DeclareState("kCreatePathClose", this, &GI2DRenderer::OnCreatePath);
    m_uiGraph.DeclareDeterministicTransition("kIdleState", "kCreatePathOpen", KeyboardButtonMap({ {'Q', kOnButtonDepressed}, {VK_CONTROL, kButtonDown} }), nullptr, 0);
    m_uiGraph.DeclareAutoTransition("kCreatePathOpen", "kCreatePathHover");
    m_uiGraph.DeclareDeterministicTransition("kCreatePathHover", "kCreatePathHover", nullptr, nullptr, kUITriggerOnMouseMove);
    m_uiGraph.DeclareDeterministicTransition("kCreatePathHover", "kCreatePathAppend", nullptr, MouseButtonMap(kMouseLButton, kOnButtonDepressed), 0);
    m_uiGraph.DeclareAutoTransition("kCreatePathAppend", "kCreatePathHover");
    m_uiGraph.DeclareDeterministicTransition("kCreatePathHover", "kCreatePathClose", KeyboardButtonMap(VK_ESCAPE, kOnButtonDepressed), nullptr, 0);
    m_uiGraph.DeclareDeterministicTransition("kCreatePathHover", "kCreatePathClose", nullptr, MouseButtonMap(kMouseRButton, kOnButtonDepressed), 0);
    m_uiGraph.DeclareAutoTransition("kCreatePathClose", "kIdleState");

    // Select/deselect path
    m_uiGraph.DeclareState("kSelectPathDragging", this, &GI2DRenderer::OnSelectPath);
    m_uiGraph.DeclareState("kSelectPathEnd", this, &GI2DRenderer::OnSelectPath);
    m_uiGraph.DeclareState("kDeselectPath", this, &GI2DRenderer::OnSelectPath);   
    m_uiGraph.DeclareNonDeterministicTransition("kIdleState", nullptr, MouseButtonMap(kMouseLButton, kOnButtonDepressed), 0, this, &GI2DRenderer::DecideOnClickState);
    m_uiGraph.DeclareDeterministicTransition("kSelectPathDragging", "kSelectPathDragging", nullptr, MouseButtonMap(kMouseLButton, kButtonDown), kUITriggerOnMouseMove);
    m_uiGraph.DeclareDeterministicTransition("kSelectPathDragging", "kSelectPathEnd", nullptr, MouseButtonMap(kMouseLButton, kOnButtonReleased), 0);
    m_uiGraph.DeclareAutoTransition("kSelectPathEnd", "kIdleState");
    m_uiGraph.DeclareDeterministicTransition("kIdleState", "kDeselectPath", nullptr, MouseButtonMap(kMouseRButton, kOnButtonDepressed), 0);
    m_uiGraph.DeclareAutoTransition("kDeselectPath", "kIdleState");

    // Move path
    m_uiGraph.DeclareState("kMovePathBegin", this, &GI2DRenderer::OnMovePath);
    m_uiGraph.DeclareState("kMovePathDragging", this, &GI2DRenderer::OnMovePath);
    //m_uiGraph.DeclareNonDeterministicTransition("kIdleState", nullptr, MouseButtonMap(kMouseLButton, kOnButtonDepressed), 0, this, &GI2DRenderer::DecideOnClickState);
    m_uiGraph.DeclareAutoTransition("kMovePathBegin", "kMovePathDragging");
    m_uiGraph.DeclareDeterministicTransition("kMovePathDragging", "kMovePathDragging", nullptr, MouseButtonMap(kMouseLButton, kButtonDown), kUITriggerOnMouseMove);
    m_uiGraph.DeclareDeterministicTransition("kMovePathDragging", "kIdleState", nullptr, MouseButtonMap(kMouseLButton, kOnButtonReleased), 0);

    // Delete path
    m_uiGraph.DeclareState("kDeletePath", this, &GI2DRenderer::OnDeletePath);    
    m_uiGraph.DeclareDeterministicTransition("kIdleState", "kDeletePath", KeyboardButtonMap({ {VK_DELETE, kOnButtonDepressed}}), nullptr, 0);
    m_uiGraph.DeclareDeterministicTransition("kIdleState", "kDeletePath", KeyboardButtonMap({ {VK_BACK, kOnButtonDepressed}}), nullptr, 0);
    m_uiGraph.DeclareAutoTransition("kDeletePath", "kIdleState");    

    m_uiGraph.Finalise();
}

GI2DRenderer::~GI2DRenderer()
{
    Destroy();
}

uint GI2DRenderer::OnIdleState(const uint& sourceStateIdx, const uint& targetStateIdx)
{
    Log::Success("Back home!");
    return kUIStateOkay;
}

uint GI2DRenderer::OnDeletePath(const uint& sourceStateIdx, const uint& targetStateIdx)
{
    if (m_objects.overlayParams.selection.numSelected != 0)
    {
        std::lock_guard <std::mutex> lock(m_resourceMutex);
        
        auto& segments = *m_objects.hostLineSegments;
        int emptyIdx = -1;
        int numDeleted = 0;
        for (int primIdx = 0; primIdx < segments.Size(); ++primIdx)
        {
            if (segments[primIdx].IsSelected())
            {
                ++numDeleted;
                if (emptyIdx == -1) { emptyIdx = primIdx; }
            }
            else if (emptyIdx >= 0)
            {
                segments[emptyIdx++] = segments[primIdx];
            }
        }

        Assert(numDeleted <= segments.Size());
        segments.Resize(segments.Size() - numDeleted);
        Log::Error("Delete!");

        m_objects.overlayParams.selection.numSelected = 0;
        SetDirtyFlags(kGI2DDirtyGeometry);
    }
    
    return kUIStateOkay;
}

uint GI2DRenderer::OnMovePath(const uint& sourceStateIdx, const uint& targetStateIdx)
{
    const std::string stateID = m_uiGraph.GetStateID(targetStateIdx);
    if (stateID == "kMovePathBegin")
    {
        m_movePath.dragAnchor = m_view.mousePos;
        Log::Write("kMovePathBegin: %s, %s", m_movePath.dragAnchor.format(), m_view.mousePos.format());
    }
    else if (stateID == "kMovePathDragging")
    {
        std::lock_guard <std::mutex> lock(m_resourceMutex);

        const vec2 delta = m_view.mousePos- m_movePath.dragAnchor;
        for (auto& segment : *m_objects.hostLineSegments)
        {
            if (segment.IsSelected())
            {
                segment += delta;
            }
        }

        Log::Write("kMovePathDragging: %s, %s", m_movePath.dragAnchor.format(), m_view.mousePos.format());

        m_movePath.dragAnchor = m_view.mousePos;
        SetDirtyFlags(kGI2DDirtyGeometry);
    }
    else
    {
        return kUIStateError;
    }

    SetDirtyFlags(kGI2DDirtyParams);
    return kUIStateOkay;
}

std::string GI2DRenderer::DecideOnClickState(const uint& sourceStateIdx, const uint& targetStateIdx)
{
    // If there are no paths selected, enter selection state. Otherwise, enter moving state.
    auto& selection = m_objects.overlayParams.selection;
    if (selection.numSelected == 0)
    {
        return "kSelectPathDragging";
    }
    else
    {
        //Assert(selection.selectedBBox.HasValidArea());
        if (selection.selectedBBox.Contains(m_view.mousePos))
        {
            return "kMovePathBegin";
        }
        else
        {
            // Deselect all the path segments
            std::lock_guard <std::mutex> lock(m_resourceMutex);
            for (auto segment : *m_objects.hostLineSegments) { segment.SetFlags(k2DPrimitiveSelected, false); }
            SetDirtyFlags(kGI2DDirtyLineSegments);
            selection.numSelected = 0;
            
            return "kSelectPathDragging";
        }
    }
}

uint GI2DRenderer::OnSelectPath(const uint& sourceStateIdx, const uint& targetStateIdx)
{
    const std::string stateID = m_uiGraph.GetStateID(targetStateIdx);
    if (stateID == "kSelectPathDragging")
    {
        auto& selection = m_objects.overlayParams.selection;
        auto& lineSegments = *m_objects.hostLineSegments;

        if (!selection.isLassoing)
        {
            // Deselect all the path segments
            for (auto& segment : lineSegments) { segment.SetFlags(k2DPrimitiveSelected, false); }
            
            selection.mouse0 = selection.mouse1 = m_view.mousePos;
            selection.isLassoing = true;
            selection.numSelected = 0;
            SetDirtyFlags(kGI2DDirtyLineSegments);
        }

        selection.lassoBBox = Rectify(BBox2f(selection.mouse0, m_view.mousePos));
        selection.selectedBBox = BBox2f::MakeInvalid();

        std::lock_guard <std::mutex> lock(m_resourceMutex);
        if (m_objects.sceneBIH->IsConstructed())
        {
            const uint lastNumSelected = selection.numSelected;
            
            auto onIntersectPrim = [&lineSegments, &selection](const uint& startIdx, const uint& endIdx, const bool isInnerNode)
            {
                // Inner nodes are tested when the bounding box envelops them completely. Hence, there's no need to do a bbox checks.
                if (isInnerNode) 
                {
                    for (int idx = startIdx; idx < endIdx; ++idx) { lineSegments[idx].SetFlags(k2DPrimitiveSelected, true); }
                    selection.numSelected += endIdx - startIdx;
                }
                else
                {
                    for (int idx = startIdx; idx < endIdx; ++idx)
                    {
                        const auto segBBox = lineSegments[idx].GetBoundingBox();
                        const bool isCaptured = selection.lassoBBox.Intersects(segBBox);
                        if (isCaptured) 
                        { 
                            selection.selectedBBox = Union(selection.selectedBBox, segBBox);
                            ++selection.numSelected; 
                        }
                        lineSegments[idx].SetFlags(k2DPrimitiveSelected, isCaptured);
                    }
                }
            };
            m_objects.sceneBIH->TestBBox(selection.lassoBBox, onIntersectPrim);

            // Only if the number of selected primitives has changed
            if (lastNumSelected != selection.numSelected)
            {
                SetDirtyFlags(kGI2DDirtyLineSegments);
            }
        }

        Log::Success("Selecting!");
    }
    else if (stateID == "kSelectPathEnd")
    {
        m_objects.overlayParams.selection.isLassoing = false; 
        SetDirtyFlags(kGI2DDirtyLineSegments);

        Log::Success("Finished!");
    }
    else if (stateID == "kDeselectPath")
    {

        Log::Success("Finished!");
    }
    else
    {
        return kUIStateError;
    }

    SetDirtyFlags(kGI2DDirtyParams);
    return kUIStateOkay;
}

uint GI2DRenderer::OnCreatePath(const uint& sourceStateIdx, const uint& targetStateIdx)
{
    const std::string stateID = m_uiGraph.GetStateID(targetStateIdx);
    if (stateID == "kCreatePathOpen")
    {
        // Record the index of the starting segment on the path 
        m_newPath.pathStartIdx = m_objects.hostLineSegments->Size() - 1;
        m_newPath.numVertices = 0;
    }
    else if (stateID == "kCreatePathHover")
    {
        if (m_newPath.numVertices >= 2)
        {
            std::lock_guard <std::mutex> lock(m_resourceMutex);
            m_objects.hostLineSegments->Back().Set(1, m_view.mousePos);

            SetDirtyFlags(kGI2DDirtyGeometry);
        }
    }
    else if (stateID == "kCreatePathAppend")
    {
        if (m_newPath.numVertices == 0)
        {
            // Create a zero-length segment that will be manipulated later
            std::lock_guard <std::mutex> lock(m_resourceMutex);
            m_objects.hostLineSegments->EmplaceBack(m_view.mousePos, m_view.mousePos, 0);
            
            m_newPath.numVertices = 2;
            SetDirtyFlags(kGI2DDirtyGeometry);
        }
        else if (m_newPath.numVertices >= 2)
        {
            // Any more and we simply reuse the last vertex on the path as the start of the next segment
            std::lock_guard <std::mutex> lock(m_resourceMutex);
            m_objects.hostLineSegments->EmplaceBack(m_objects.hostLineSegments->Back()[1], m_view.mousePos, 0);
            
            m_newPath.numVertices++;
            SetDirtyFlags(kGI2DDirtyGeometry);
        }
    }
    else if (stateID == "kCreatePathClose")
    {

    }
    else
    {
        return kUIStateError;
    }

    return kUIStateOkay;
}

std::shared_ptr<RendererInterface> GI2DRenderer::Instantiate()
{
    return std::make_shared<GI2DRenderer>();
}

void GI2DRenderer::RebuildBIH()
{
    // Synchronise the segments
    Cuda::Host::Vector<GI2D::LineSegment>& segments = *m_objects.hostLineSegments;
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

    //SetDirtyFlags(kGI2DDirtyLineSegments);
}

void GI2DRenderer::OnInitialise()
{
    m_view.trans = vec2(0.f);
    m_view.scale = 1.0f;
    m_view.rotate = 0.0;
    m_view.matrix = ConstructViewMatrix(m_view.trans, m_view.rotate, m_view.scale) * m_clientToNormMatrix;
    m_view.zoomSpeed = 10.0f;

    //m_primitiveContainer.Create(m_renderStream);

    m_objects.hostLineSegments = CreateAsset<Cuda::Host::Vector<GI2D::LineSegment>>("id_lineSegments", kVectorHostAlloc, m_renderStream);
    m_objects.sceneBIH = CreateAsset<GI2D::Host::BIH2DAsset>("id_gi2DBIH");

    m_objects.overlayRenderer = CreateAsset<GI2D::Host::Overlay>("id_gi2DOverlay", m_objects.sceneBIH, m_objects.hostLineSegments, m_clientWidth, m_clientHeight, m_renderStream);
    m_objects.pathTracer = CreateAsset<GI2D::Host::PathTracer>("id_gi2DPathTracer", m_objects.sceneBIH, m_objects.hostLineSegments, m_clientWidth, m_clientHeight, 2, m_renderStream);

    SetDirtyFlags(kGI2DDirtyAll);
}

void GI2DRenderer::OnDestroy()
{
    m_objects.overlayRenderer.DestroyAsset();
    m_objects.sceneBIH.DestroyAsset();
    m_objects.hostLineSegments.DestroyAsset();
}

void GI2DRenderer::OnRender()
{
    //std::this_thread::sleep_for(std::chrono::milliseconds(50));
    //Log::Write("Tick");

    if (m_dirtyFlags)
    {
        std::lock_guard <std::mutex> lock(m_resourceMutex);

        //Log::Warning("Dirty: %i", m_dirtyFlags);

        if (m_dirtyFlags & kGI2DDirtyParams)
        {
            const float dPdXY = length(vec2(m_view.matrix.i00, m_view.matrix.i10));
            
            m_objects.overlayParams.mousePosView = m_view.mousePos;
            m_objects.viewTransform = GI2D::ViewTransform(m_view.matrix, m_view.trans, m_view.rotate, m_view.scale, m_view.mousePos, dPdXY);            
            
            m_objects.overlayParams.view = m_objects.pathTracerParams.view = m_objects.viewTransform;
            m_objects.overlayParams.sceneBounds = m_objects.pathTracerParams.sceneBounds = BBox2f(vec2(-0.5f), vec2(0.5f));
            m_objects.pathTracerParams.frameIdx = m_frameIdx;

            m_objects.overlayRenderer->SetParams(m_objects.overlayParams);
            m_objects.pathTracer->SetParams(m_objects.pathTracerParams);
        }

        if (m_dirtyFlags & kGI2DDirtyBIH)
        {
            RebuildBIH();
        }
        else if (m_dirtyFlags & kGI2DDirtyLineSegments)
        {
            m_objects.hostLineSegments->Synchronise(kVectorSyncUpload);
        }

        m_objects.pathTracer->SetDirty();
        m_objects.overlayRenderer->SetDirty();

        ClearDirtyFlags(kGI2DDirtyAll);
    }

    // Render the pass
    m_objects.pathTracer->Render();
    m_objects.overlayRenderer->Render();

    // If a blit is in progress, skip the composite step entirely.
    // TODO: Make this respond intelligently to frame rate. If the CUDA renderer is running at a lower FPS than the D3D renderer then it should wait rather than
    // than skipping frames like this.
    //m_renderSemaphore.Wait(kRenderManagerD3DBlitFinished, kRenderManagerCompInProgress);
    if (!m_renderSemaphore.Try(kRenderManagerD3DBlitFinished, kRenderManagerCompInProgress, false)) { return; }
    
    m_objects.pathTracer->Composite(m_compositeImage);
    m_objects.overlayRenderer->Composite(m_compositeImage);

    m_renderSemaphore.Try(kRenderManagerCompInProgress, kRenderManagerCompFinished, true);
}

void GI2DRenderer::OnKey(const uint code, const bool isSysKey, const bool isDown)
{

}

void GI2DRenderer::OnMouseButton(const uint code, const bool isDown)
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

mat3 GI2DRenderer::ConstructViewMatrix(const vec2& trans, const float rotate, const float scale) const
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

void GI2DRenderer::OnMouseMove()
{
    // Dragging?
    if (IsMouseButtonDown(kMouseMButton))
    {
        OnViewChange();
    }

    {
        std::lock_guard <std::mutex> lock(m_resourceMutex);
        m_view.mousePos = m_view.matrix * vec2(m_mouse.pos);
    }
}

void GI2DRenderer::OnViewChange()
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
        m_view.matrix = ConstructViewMatrix(m_view.transAnchor, m_view.rotate, m_view.scale) * m_clientToNormMatrix;
        const vec2 dragDelta = (m_view.matrix * vec2(m_view.dragAnchor)) - (m_view.matrix * vec2(m_mouse.pos));
        m_view.trans = m_view.transAnchor + dragDelta;

        //Log::Write("Trans: %s", m_view.trans.format());
    }

    // Update the parameters in the overlay renderer
    {
        std::lock_guard <std::mutex> lock(m_resourceMutex);
        m_view.matrix = ConstructViewMatrix(m_view.trans, m_view.rotate, m_view.scale) * m_clientToNormMatrix;

        // Mark the scene as dirty
        SetDirtyFlags(kGI2DDirtyParams);
    }
}

void GI2DRenderer::OnMouseWheel()
{

}

void GI2DRenderer::OnResizeClient()
{
}