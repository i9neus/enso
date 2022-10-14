#include "ObjectDebugger.cuh"

#include "kernels/math/CudaColourUtils.cuh"

#include "tracables/primitives/LineSegment.cuh"
#include "GenericIntersector.cuh"
#include "kernels/math/CudaColourUtils.cuh"
#include "BIH2DAsset.cuh"

#include "tracables/Curve.cuh"

#include "kernels/DeviceAllocator.cuh"

namespace GI2D
{
    __device__ bool SceneObjectInterface2::EvaluateControlHandles(const vec2& pWorld, const UIViewCtx& viewCtx, vec4& L) const
    {
        // Draw the bounding box
        /*if (m_params.objectBBox.PointOnPerimiter(p, m_params.viewCtx.dPdXY * 2.f))
        {
            L = vec4(kOne, 1.0f);
            return true;
        }*/

        // Draw the control handles
        if (m_handleInnerBBox.IsValid() && !m_handleInnerBBox.Contains(pWorld))
        {
            //L = Blend(L, kOne, 0.2f);

            /*vec2 hp = 2.0f * (p - m_handleInnerBBox.lower) / (m_handleInnerBBox.Dimensions() - vec2(m_params.viewCtx.dPdXY * 10.0f));

            if (fract(hp.x) < 0.1f && fract(hp.y) < 0.1f)
            {
                 L = vec4(1.0f);
                 return true;
            }*/
        }

        return false;
    }

    __host__ Host::SceneObject2::SceneObject2(const std::string& id) :
        RenderObject(id),
        m_dirtyFlags(kGI2DDirtyAll),
        m_isFinalised(false),
        cu_deviceSceneObjectInterface2(nullptr)
    {
    }

    __host__ uint Host::SceneObject2::OnSelect(const bool isSelected)
    {
        if (SetGenericFlags(m_attrFlags, uint(kSceneObjectSelected), isSelected))
        {
            SetDirtyFlags(kGI2DDirtyUI, true);
        }
        return m_dirtyFlags;
    }

    __host__ uint Host::SceneObject2::OnMove(const std::string& stateID, const UIViewCtx& viewCtx)
    {
        if (stateID == "kMoveSceneObjectBegin")
        {
            m_onMove.dragAnchor = viewCtx.mousePos;
            m_onMove.isDragging = true;
            Log::Error("kMoveSceneObjectBegin");
        }
        else if (stateID == "kMoveSceneObjectDragging")
        {
            Assert(m_onMove.isDragging);

            const vec2 delta = viewCtx.mousePos - m_onMove.dragAnchor;
            m_onMove.dragAnchor = viewCtx.mousePos;
            m_transform.trans += delta;
            m_worldBBox += delta;

            // The geometry internal to this object hasn't changed, but it will affect the 
            Log::Warning("kMoveSceneObjectDragging");
            SetDirtyFlags(kGI2DDirtyBVH);
        }
        else if (stateID == "kMoveSceneObjectEnd")
        {
            m_onMove.isDragging = false;
            Log::Success("kMoveSceneObjectEnd");
        }

        return m_dirtyFlags;
    }

    __host__ __device__ bool TracableInterface2::IntersectBBox(const BBox2f& bBox) const
    {
        return bBox.Intersects(m_objectBBox);
    }

    __host__ __device__ bool Device::Curve2::IntersectRay(Ray2D& rayWorld, HitCtx2D& hitWorld) const
    {
        assert(m_bih && m_lineSegments);

        RayRange2D range;
        if (!IntersectRayBBox(rayWorld, m_worldBBox, range) || range.tNear > hitWorld.tFar) { return false; }

        RayBasic2D& rayObject = ToObjectSpace(rayWorld);
        HitCtx2D hitObject;

        auto onIntersect = [&](const uint* startEndIdx, RayRange2D& rangeTree)
        {
            for (int primIdx = startEndIdx[0]; primIdx < startEndIdx[1]; ++primIdx)
            {
                if ((*m_lineSegments)[primIdx].IntersectRay(rayObject, hitObject) && hitObject.tFar < rangeTree.tFar && hitObject.tFar < hitWorld.tFar)
                {
                    rangeTree.tFar = hitObject.tFar;
                }
            }
        };
        m_bih->TestRay(rayObject, range.tFar, onIntersect);

        if (hitObject.tFar < hitWorld.tFar)
        {
            hitWorld.tFar = hitObject.tFar;
            return true;
        }

        return false;
    }

    /*__host__ __device__ bool Device::Curve2::InteresectPoint(const vec2& p, const float& thickness) const
    {
    }*/

    /*__host__ __device__ vec2 Device::Curve2::PerpendicularPoint(const vec2& p) const
    {
    }*/

    __device__ bool Device::Curve2::EvaluatePrimitives(const vec2& pWorld, const UIViewCtx& viewCtx, vec4& L) const
    {
        vec4 LPrim(0.0f);
        const vec2 pLocal = pWorld - m_transform.trans;

        if (!m_bih) { return false; }

        m_bih->TestPoint(pLocal, [&, this](const uint* idxRange)
            {
                for (int idx = idxRange[0]; idx < idxRange[1]; ++idx)
                {
                    const auto& segment = (*m_lineSegments)[idx];
                    const float line = segment.Evaluate(pLocal, viewCtx.dPdXY);
                    if (line > 0.f)
                    {
                        LPrim = Blend(LPrim, segment.IsSelected() ? vec3(1.0f, 0.1f, 0.0f) : kOne, line);
                    }
                }
            });

        if (LPrim.w > 0.0f)
        {
            L = Blend(L, LPrim);
            return true;
        }
        return false;
    }

    __host__ Host::Curve2::Curve2(const std::string& id) :
        Tracable2(id),
        cu_deviceInstance(nullptr)
    {
        Log::Success("Host::Curve2::Curve");

        constexpr uint kMinTreePrims = 3;

        m_hostBIH = CreateChildAsset<GI2D::Host::BIH2DAsset>("bih", kMinTreePrims);
        m_hostLineSegments = CreateChildAsset<Cuda::Host::Vector<LineSegment>>("lineSegments", kVectorHostAlloc, nullptr);

        cu_deviceInstance = InstantiateOnDevice<Device::Curve2>();
        cu_deviceTracableInterface2 = StaticCastOnDevice<TracableInterface2>(cu_deviceInstance);

        m_deviceObjects.m_bih = m_hostBIH->GetDeviceInstance();
        m_deviceObjects.m_lineSegments = m_hostLineSegments->GetDeviceInterface();

        // Set the host parameters so we can query the primitive on the host
        m_bih = static_cast<BIH2D<BIH2DFullNode>*>(m_hostBIH.get());
        m_lineSegments = static_cast<Cuda::VectorInterface<GI2D::LineSegment>*>(m_hostLineSegments.get());

        Synchronise(kSyncObjects);
    }

    __host__ Host::Curve2::~Curve2()
    {
        Log::Error("Host::Curve2::~Curve");
        OnDestroyAsset();
    }

    __host__ void Host::Curve2::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceInstance);

        m_hostBIH.DestroyAsset();
        m_hostLineSegments.DestroyAsset();
    }

    __host__ void Host::Curve2::Synchronise(const int syncType)
    {
        Tracable2::Synchronise(cu_deviceInstance, syncType);

        if (syncType == kSyncObjects)
        {
            SynchroniseObjects3<CurveObjects2>(cu_deviceInstance, m_deviceObjects);
        }
    }

    __host__ uint Host::Curve2::OnCreate(const std::string& stateID, const UIViewCtx& viewCtx)
    {
        const vec2 mousePosLocal = viewCtx.mousePos - m_transform.trans;
        if (stateID == "kCreateSceneObjectOpen")
        {
            //m_transform.trans = viewCtx.mousePos;

            Log::Success("Opened path %s", GetAssetID());
        }
        else if (stateID == "kCreateSceneObjectHover")
        {
            if (!m_hostLineSegments->IsEmpty())
            {
                m_hostLineSegments->Back().Set(1, mousePosLocal);
                SetDirtyFlags(kGI2DDirtyBVH);
            }
        }
        else if (stateID == "kCreateSceneObjectAppend")
        {
            const vec3 colour = Hue(PseudoRNG(HashOf(m_hostLineSegments->Size())).Rand<0>());

            if (m_hostLineSegments->IsEmpty())
            {
                // Create a zero-length segment that will be manipulated later
                m_hostLineSegments->EmplaceBack(mousePosLocal, mousePosLocal, 0, colour);
            }
            else
            {
                // Any more and we simply reuse the last vertex on the path as the start of the next segment
                m_hostLineSegments->EmplaceBack(m_hostLineSegments->Back()[1], mousePosLocal, 0, colour);
            }

            SetDirtyFlags(kGI2DDirtyBVH);
        }
        else if (stateID == "kCreateSceneObjectClose")
        {
            // Delete the floating segment when closing the path
            if (!m_hostLineSegments->IsEmpty())
            {
                m_hostLineSegments->PopBack();
            }

            Log::Warning("Closed path %s", GetAssetID());
            SetDirtyFlags(kGI2DDirtyBVH);
        }
        else
        {
            AssertMsg(false, "Invalid state");
        }

        return m_dirtyFlags;
    }

    __host__ bool Host::Curve2::IsConstructed() const
    {
        return !m_hostLineSegments->IsEmpty() && m_hostBIH->IsConstructed();
    }

    __host__ bool Host::Curve2::Finalise()
    {
        m_isFinalised = true;

        return IsConstructed();
    }

    __host__ bool Host::Curve2::Rebuild(const uint parentFlags, const UIViewCtx& viewCtx)
    {
        if (!m_dirtyFlags) { return IsConstructed(); }

        bool resyncParams = false;

        // If the geometry has changed, rebuild the BIH
        if (m_dirtyFlags & kGI2DDirtyBVH)
        {
            // Sync the line segments
            auto& segments = *m_hostLineSegments;
            segments.Synchronise(kVectorSyncUpload);

            // Create a segment list ready for building
            // TODO: It's probably faster if we build on the already-sorted index list
            auto& primIdxs = m_hostBIH->GetPrimitiveIndices();
            primIdxs.resize(segments.Size());
            for (uint idx = 0; idx < primIdxs.size(); ++idx) { primIdxs[idx] = idx; }

            // Construct the BIH
            std::function<BBox2f(uint)> getPrimitiveBBox = [&segments](const uint& idx) -> BBox2f
            {
                return Grow(segments[idx].GetBoundingBox(), 0.001f);
            };
            m_hostBIH->Build(getPrimitiveBBox);

            // Update the tracable bounding boxes
            //m_objectBBox = m_hostBIH->GetBoundingBox();
            //m_worldBBox = m_objectBBox + m_transform.trans;
            //Log::Write("  - Rebuilt curve %s BIH: %s", GetAssetID(), GetObjectSpaceBoundingBox().Format()); 

            resyncParams = true;
        }

        if (m_dirtyFlags & kGI2DDirtyTransforms)
        {
            resyncParams = true;
        }

        if (resyncParams) { Synchronise(kSyncParams); }

        ClearDirtyFlags();

        return IsConstructed();
    }
















    __host__ __device__ OverlayParams2::OverlayParams2()
    {
        m_gridCtx.show = true;
        m_gridCtx.lineAlpha = 0.0;
        m_gridCtx.majorLineSpacing = 1.0f;
        m_gridCtx.majorLineSpacing = 1.0f;
    }

    __device__ Device::Overlay2::Overlay2()
    {
    }

    __device__ void Device::Overlay2::Composite(Cuda::Device::ImageRGBA* deviceOutputImage)
    {
        assert(deviceOutputImage);

        // TODO: Make alpha compositing a generic operation inside the Image class.
        const ivec2 xyScreen = kKernelPos<ivec2>();
        if (xyScreen.x >= 0 && xyScreen.x < m_accumBuffer->Width() && xyScreen.y >= 0 && xyScreen.y < m_accumBuffer->Height())
        {
            deviceOutputImage->Blend(xyScreen, m_accumBuffer->At(xyScreen));
            //vec4& target = deviceOutputImage->At(xyScreen);
            //target = Blend(target, m_accumBuffer->At(xyScreen));
            //target.xyz += m_accumBuffer->At(xyScreen).xyz;
        }
    }
    DEFINE_KERNEL_PASSTHROUGH_ARGS(Composite);

    __device__ void Device::Overlay2::Render()
    {
        assert(m_accumBuffer);

        printf("!!!!!!!!!!!\nbih: 0x%llx\ntracables: 0x%llx\n", m_bih, m_tracables);

        const ivec2 xyScreen = kKernelPos<ivec2>();
        if (xyScreen.x < 0 || xyScreen.x >= m_accumBuffer->Width() || xyScreen.y < 0 || xyScreen.y >= m_accumBuffer->Height()) { return; }

        // Transform from screen space to view space
        const vec2 xyView = m_viewCtx.transform.matrix * vec2(xyScreen);

        //m_accumBuffer->At(xyScreen) = vec4(xyView, 0.0f, 1.0f);
        //return;

        vec4 L(0.0f, 0.0f, 0.0f, 0.0f);

        if (!m_viewCtx.sceneBounds.Contains(xyView))
        {
            L = vec4(0.0f);
        }
        else if (m_gridCtx.show)
        {
            // Draw the grid
            vec2 xyGrid = fract(xyView / vec2(m_gridCtx.majorLineSpacing)) * sign(xyView);
            if (cwiseMin(xyGrid) < m_viewCtx.dPdXY / m_gridCtx.majorLineSpacing * mix(1.0f, 3.0f, m_gridCtx.lineAlpha))
            {
                L = Blend(L, kOne, 0.5 * (1 - m_gridCtx.lineAlpha));
            }
            xyGrid = fract(xyView / vec2(m_gridCtx.minorLineSpacing)) * sign(xyView);
            if (cwiseMin(xyGrid) < m_viewCtx.dPdXY / m_gridCtx.minorLineSpacing * 1.5f)
            {
                L = Blend(L, kOne, 0.5 * m_gridCtx.lineAlpha);
            }
        }

        // Draw the tracables
        if (m_bih && m_tracables)
        {
            const VectorInterface<TracableInterface2*>& tracables = *(m_tracables);

            auto onPointIntersectLeaf = [&, this](const uint* idxRange) -> void
            {
                for (int idx = idxRange[0]; idx < idxRange[1]; ++idx)
                {
                    assert(idx < tracables.Size());
                    assert(tracables[idx]);

                    const auto& tracable = *tracables[idx];
                    vec4 LTracable;
                    if (tracable.EvaluateOverlay(xyView, m_viewCtx, LTracable))
                    {
                        L = Blend(L, LTracable);
                    }

                    if (tracable.GetWorldSpaceBoundingBox().PointOnPerimiter(xyView, m_viewCtx.dPdXY)) L = vec4(kRed, 1.0f);
                }
            };
            //m_bih->TestPoint(xyView, onPointIntersectLeaf);
        }

        // Draw the widgets
        /*if (m_inspectors)
        {
            for (int idx = 0; idx < m_inspectors->Size(); ++idx)
            {
                vec4 LWidget;
                if ((*m_inspectors)[idx]->EvaluateOverlay(xyView, m_viewCtx, LWidget))
                {
                    L = Blend(L, LWidget);
                }
            }
        }*/

        // Draw the lasso 
        if (m_selectionCtx.isLassoing && m_selectionCtx.lassoBBox.PointOnPerimiter(xyView, m_viewCtx.dPdXY * 2.f)) { L = vec4(kRed, 1.0f); }

        // Draw the selected object's bounding box
        if (m_selectionCtx.numSelected > 0. && m_selectionCtx.selectedBBox.PointOnPerimiter(xyView, m_viewCtx.dPdXY * 2.f)) { L = vec4(kGreen, 1.0f); }

        m_accumBuffer->At(xyScreen) = L;
    }
    DEFINE_KERNEL_PASSTHROUGH(Render);

    Host::Overlay2::Overlay2(const std::string& id, AssetHandle<Host::BIH2DAsset>& bih, AssetHandle<TracableContainer2>& tracables, AssetHandle<InspectorContainer2>& inspectors,
        const uint width, const uint height, cudaStream_t renderStream) :
        UILayer2(id, bih, tracables),
        m_hostInspectors(inspectors)
    {
        // Create some Cuda objects
        m_hostAccumBuffer = CreateChildAsset<Cuda::Host::ImageRGBW>("accumBuffer", width, height, renderStream);

        if(m_hostBIH) m_deviceObjects.m_bih = m_hostBIH->GetDeviceInstance();
        m_deviceObjects.m_tracables = m_hostTracables->GetDeviceInterface();
        m_deviceObjects.m_accumBuffer = m_hostAccumBuffer->GetDeviceInstance();
        if(m_hostInspectors) m_deviceObjects.m_inspectors = m_hostInspectors->GetDeviceInterface();

        cu_deviceData = InstantiateOnDevice<Device::Overlay2>();

        Synchronise(kSyncObjects);
    }

    Host::Overlay2::~Overlay2()
    {
        OnDestroyAsset();
    }

    __host__ void Host::Overlay2::Synchronise(const int syncType)
    {
        UILayer2::Synchronise(cu_deviceData, syncType);

        if (syncType & kSyncObjects) { SynchroniseObjects2<OverlayObjects2>(cu_deviceData, m_deviceObjects); }
        if (syncType & kSyncParams) { SynchroniseObjects2<OverlayParams2>(cu_deviceData, *this); }
    }

    __host__ void Host::Overlay2::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceData);
        m_hostAccumBuffer.DestroyAsset();
    }

    __host__ void Host::Overlay2::Render()
    {
        if (!m_dirtyFlags) { return; }

        dim3 blockSize, gridSize;
        KernelParamsFromImage(m_hostAccumBuffer, blockSize, gridSize);

        KernelRender << < 1, 1>> > (cu_deviceData);
        IsOk(cudaDeviceSynchronize());

        m_dirtyFlags = 0;
    }

    __host__ void Host::Overlay2::Composite(AssetHandle<Cuda::Host::ImageRGBA>& hostOutputImage) const
    {
        dim3 blockSize, gridSize;
        KernelParamsFromImage(m_hostAccumBuffer, blockSize, gridSize);

        KernelComposite << < gridSize, blockSize, 0, m_hostStream >> > (cu_deviceData, hostOutputImage->GetDeviceInstance());
        IsOk(cudaDeviceSynchronize());
    }

    /*__host__ void Host::Overlay2::TraceRay()
    {
        const auto& tracables = *m_hostTracables;
        Ray2D ray(vec2(0.0f), normalize(m_viewCtx.mousePos));
        HitCtx2D hit;

        auto onIntersect = [&](const uint* primRange, RayRange2D& range)
        {
            for (uint idx = primRange[0]; idx < primRange[1]; ++idx)
            {
                if (tracables[idx]->IntersectRay(ray, hit))
                {
                    if (hit.tFar < range.tFar)
                    {
                        range.tFar = hit.tFar;
                    }
                }
            }
        };
        m_hostBIH->TestRay(ray, kFltMax, onIntersect);
    }*/

    __host__ void Host::Overlay2::Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx)
    {
        UILayer2::Rebuild(dirtyFlags, viewCtx, selectionCtx);

        if (!m_dirtyFlags) { return; }

        // Calculate some values for the guide grid
        const float logScale = std::log10(m_viewCtx.transform.scale);
        constexpr float kGridScale = 0.05f;
        m_gridCtx.majorLineSpacing = kGridScale * std::pow(10.0f, std::ceil(logScale));
        m_gridCtx.minorLineSpacing = kGridScale * std::pow(10.0f, std::floor(logScale));
        m_gridCtx.lineAlpha = 1 - (logScale - std::floor(logScale));
        m_gridCtx.show = true;
        m_selectionCtx.lassoBBox.Rectify();
        m_selectionCtx.selectedBBox.Rectify();

        // Upload to the device
        Synchronise(kSyncParams);
    }

















    __global__ void KernelTest(BIH2D<BIH2DFullNode>* bih, VectorInterface<TracableInterface2*>* tracablesPtr, VectorInterface<TracableInterface2*>* inspectorsPtr)
    {
        printf("%i bytes\n", sizeof(VectorInterface<TracableInterface2*>*));
        printf("@@@@@@@@@@\nbih: 0x%llx\ntracables: 0x%llx\n", bih, tracablesPtr);
        void* bad = (void*)(0xffffffffffffffff);
        printf("bad: 0x%llx\n", bad);
        
        vec2 xyView;
        UIViewCtx viewCtx;
        UIGridCtx grid;
        vec4 L;
         
        // Draw the tracables
        if (bih && tracablesPtr)
        {
            const VectorInterface<TracableInterface2*>& tracables = *tracablesPtr;

            printf("Tracables\n");

            auto onPointIntersectLeaf = [&](const uint* idxRange) -> void
            {
                for (int idx = idxRange[0]; idx < idxRange[1]; ++idx)
                {
                    assert(idx < tracables.Size());
                    assert(tracables[idx]);

                    const auto& tracable = *tracables[idx];
                    vec4 LTracable;
                    if (tracable.EvaluateOverlay(xyView, viewCtx, LTracable))
                    {
                        L = Blend(L, LTracable);
                    }

                    if (tracable.GetWorldSpaceBoundingBox().PointOnPerimiter(xyView, viewCtx.dPdXY)) L = vec4(kRed, 1.0f);
                }
            };

            // Clone the BIH class and check that it's working properly by removing features until the crashes stop
            bih->TestPoint(xyView, onPointIntersectLeaf);
        }

        // Draw the widgets
       if (inspectorsPtr)
       {
           printf("Inspectors\n");
           
           for (int idx = 0; idx < inspectorsPtr->Size(); ++idx)
           {
               vec4 LWidget;
               if ((*inspectorsPtr)[idx]->EvaluateOverlay(xyView, viewCtx, LWidget))
               {
                   L = Blend(L, LWidget);
               }
           }
       }

       {
           printf("Grid\n");

           // Draw the grid
           vec2 xyGrid = fract(xyView / vec2(grid.majorLineSpacing)) * sign(xyView);
           if (cwiseMin(xyGrid) < viewCtx.dPdXY / grid.majorLineSpacing * mix(1.0f, 3.0f, grid.lineAlpha))
           {
               L = Blend(L, kOne, 0.5 * (1 - grid.lineAlpha));
           }
           xyGrid = fract(xyView / vec2(grid.minorLineSpacing)) * sign(xyView);
           if (cwiseMin(xyGrid) < viewCtx.dPdXY / grid.minorLineSpacing * 1.5f)
           {
               L = Blend(L, kOne, 0.5 * grid.lineAlpha);
           }
       }


       printf("End\n");
    }

    __host__ void      InvokeDebugger()
    {
        AssetHandle<InspectorContainer2> inspectors = CreateAsset<TracableContainer2>("test/inspectors", kVectorHostAlloc, nullptr);
        AssetHandle<TracableContainer2> tracables = CreateAsset<TracableContainer2>("test/tracables", kVectorHostAlloc, nullptr);
        AssetHandle<Host::Curve2> curve = CreateAsset<Host::Curve2>("test/curve");
        AssetHandle<Host::BIH2DAsset> bih = CreateAsset<Host::BIH2DAsset>("test/bih", 1);

        //tracables->PushBack(curve);

        AssetHandle<Host::Overlay2> overlayRenderer = CreateAsset<Host::Overlay2>("test/overlay", bih, tracables, inspectors, 100, 100, nullptr);

        tracables->Clear();
        tracables->Synchronise(kVectorSyncUpload);

        Log::Warning("tracables: 0x%x", tracables->GetDeviceInterface());
        Log::Warning("bih: 0x%x", bih->GetDeviceInstance());

        Log::Error("Invoking...");
        KernelTest << <1, 1 >> > (bih->GetDeviceInterface(), tracables->GetDeviceInterface(), inspectors->GetDeviceInterface());
        IsOk(cudaDeviceSynchronize());

        Log::Error("Rendering...");
        UIViewCtx view;
        UISelectionCtx selection;
        //overlayRenderer->Rebuild(0xffffffff, view, selection);
        overlayRenderer->Render();

        tracables.DestroyAsset();
        curve.DestroyAsset();
        overlayRenderer.DestroyAsset();
        bih.DestroyAsset();
    }

    
}