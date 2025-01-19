#define CUDA_DEVICE_GLOBAL_ASSERTS

#include "NNanoSDF.cuh"
#include "Beziers.h"
#include "core/math/ColourUtils.cuh"
#include "core/math/Hash.cuh"
#include "core/assets/AssetContainer.cuh"
#include "core/containers/Vector.cuh"
#include "core/assets/GenericObjectContainer.cuh"
#include "core/2d/Transform2D.cuh"
#include "core/2d/primitives/QuadraticSpline.cuh"
#include "core/2d/primitives/Ellipse.cuh"
#include "core/2d/bih/BIH2DAsset.cuh"

#include "io/json/JsonUtils.h"
//#include "core/AccumulationBuffer.cuh"

namespace Enso
{
    __host__ __device__ NNanoSDFParams::NNanoSDFParams()
    {
        viewport.dims = ivec2(0);
    }

    __host__ __device__ void NNanoSDFParams::Validate() const
    {
        CudaAssert(viewport.dims.x != 0 && viewport.dims.y != 0);
    }

    __host__ __device__ void Device::NNanoSDF::Synchronise(const NNanoSDFParams& params)
    {
        m_params = params;
    }

    __device__ void Device::NNanoSDF::Synchronise(const NNanoSDFObjects& objects)
    {
        objects.Validate();
        m_objects = objects;        
    }

    __host__ __device__ bool Device::NNanoSDF::IsClickablePoint(const UIViewCtx& viewCtx) const
    {
        return GetWorldBBox().Contains(viewCtx.mousePos);
    }

    __host__ __device__ vec4 Device::NNanoSDF::EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx, const bool isMouseTest) const
    {
        if (!GetWorldBBox().Contains(pWorld) || !m_objects.bih) { return vec4(0.0f); }        

#ifdef __CUDA_ARCH__
        const vec2 pObject = ToObjectSpace(pWorld);
        const vec2 pScreen = (pObject - m_params.viewport.objectBounds.lower) / m_params.viewport.objectBounds.Dimensions();
        const vec2 pView = TransformNormalisedScreenToView(pScreen, m_params.viewport.dims);
        
        vec4 L(0.0f);
        const OverlayCtx overlayCtx = OverlayCtx::MakeStroke(viewCtx, vec4(1.f), 3.f);
        const auto& splineList = *m_objects.splines;
        auto onLeaf = [&](const uint* idxRange, const uint* primIdxs) -> bool
        {
            for (int idx = idxRange[0]; idx < idxRange[1]; ++idx)
            {
                const auto& spline = splineList[primIdxs[idx]];
                const vec4 line = spline.EvaluateOverlay(pView, overlayCtx);
                if (line.w > 0.f) { L = Blend(L, line); }
            }

            return false;
        };
        /*auto onInner = [&, this](const BBox2f& bBox, const int depth) -> void
        {
            if (bBox.PointOnPerimiter(pView, viewCtx.dPdXY * 10.))
                L = vec4(kRed, 1.0f);
        };*/
        m_objects.bih->TestPoint(pView, onLeaf, nullptr);

        /*for (int idx = 0; idx < splineList.size(); ++idx)
        {
            const auto& spline = splineList[idx];
            //const vec4 line = spline.EvaluateOverlay(pView, overlayCtx);
            if (spline.GetBoundingBox().PointOnPerimiter(pView, viewCtx.dPdXY * 10.))
                L = vec4(Hue(float(idx) / 10), 1.0);
            //if (line.w > 0.f) { L = Blend(L, line); }
        }*/

        return L;
#else
        return vec4(1.);
#endif
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    __host__ AssetHandle<Host::GenericObject> Host::NNanoSDF::Instantiate(const std::string& id, const Host::Asset& parentAsset, const AssetHandle<const Host::GenericObjectContainer>& genericObjects)
    {
        return AssetAllocator::CreateChildAsset<Host::NNanoSDF>(parentAsset, id, genericObjects);
    }

    __host__ Host::NNanoSDF::NNanoSDF(const Asset::InitCtx& initCtx, const AssetHandle<const Host::GenericObjectContainer>& genericObjects) :
        DrawableObject(initCtx, &m_hostInstance),
        cu_deviceInstance(AssetAllocator::InstantiateOnDevice<Device::NNanoSDF>(*this))
    {
        DrawableObject::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::DrawableObject>(cu_deviceInstance));

        constexpr int kViewportWidth = 1200;
        constexpr int kViewportHeight = 675;

        const vec2 boundHalf = 0.25 * ((kViewportHeight > kViewportWidth) ?
            vec2(1.f, float(kViewportHeight) / float(kViewportWidth)) :
            vec2(float(kViewportWidth) / float(kViewportHeight), 1.f));

        m_params.viewport.dims = ivec2(kViewportWidth, kViewportHeight);
        m_params.viewport.objectBounds = BBox2f(-boundHalf, boundHalf);

        constexpr uint kMinTreePrims = 0;
        m_hostBIH = AssetAllocator::CreateChildAsset<Host::BIH2DAsset>(*this, "bih", kMinTreePrims);
        m_hostSplines = AssetAllocator::CreateChildAsset<Host::Vector<QuadraticSpline>>(*this, "splines");

        // Set the host parameters so we can query the primitive on the host
        m_deviceObjects.bih = m_hostBIH->GetDeviceInstance();
        m_deviceObjects.splines = m_hostSplines->GetDeviceInstance();

        // Populate the splines from the embedded data
        auto& splines = *m_hostSplines;       
        using namespace FlowBeziers;
        splines.resize(kNumSplines);
        for (int i = 0; i < kNumSplines; ++i)
        {
            splines[i] = QuadraticSpline(kPoints[i * 2], kPoints[i * 2 + 1]);
        }
        splines.Upload();

        // Create a primitive list ready for building
        // TODO: It's probably faster if we build on the already-sorted index list
        auto& primIdxs = m_hostBIH->GetPrimitiveIndices();
        primIdxs.resize(splines.size());
        for (uint idx = 0; idx < primIdxs.size(); ++idx) { primIdxs[idx] = idx; }

        // Construct the BIH
        std::function<BBox2f(uint)> getPrimitiveBBox = [&splines](const uint& idx) -> BBox2f
        {
            return Scale(splines[idx].GetBoundingBox(), 1.01f);
        };
        m_hostBIH->Build(getPrimitiveBBox, true);

        Synchronise(kSyncObjects | kSyncParams);

        Cascade({ kDirtySceneObjectChanged });
    }

    __host__ Host::NNanoSDF::~NNanoSDF() noexcept
    {
        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance); 

        m_hostBIH.DestroyAsset();
        m_hostSplines.DestroyAsset();
    }

    __host__ void Host::NNanoSDF::OnSynchroniseDrawableObject(const uint syncFlags)
    {
        // Only sync the objects if a SceneContainer has been bound
        if (syncFlags & kSyncObjects)
        {
            SynchroniseObjects<Device::NNanoSDF>(cu_deviceInstance, m_deviceObjects);
        }
        if (syncFlags & kSyncParams)
        {
            SynchroniseObjects<Device::NNanoSDF>(cu_deviceInstance, m_params);
            m_hostInstance.Synchronise(m_params);
        }
    }

    __host__ void Host::NNanoSDF::Bind(GenericObjectContainer& objects) {}
   
    __host__ void Host::NNanoSDF::Clear()
    {
        Synchronise(kSyncParams);
    }

    __host__ bool Host::NNanoSDF::OnCreateDrawableObject(const std::string& stateID, const UIViewCtx& viewCtx, const vec2& mousePosObject)
    {
        if (stateID == "kCreateDrawableObjectOpen" || stateID == "kCreateDrawableObjectHover")
        {
            m_isConstructed = true;
            m_isFinalised = true;
            if (stateID == "kCreateDrawableObjectOpen") { Log::Success("Opened path tracer %s", GetAssetID()); }

            return true;
        }
        else if (stateID == "kCreateDrawableObjectAppend")
        {
            m_isFinalised = true;
            return true;
        }

        return false;
    }

    __host__ bool Host::NNanoSDF::OnRebuildDrawableObject()
    {
        /*m_scene = m_componentContainer->GenericObjects().FindFirstOfType<Host::SceneContainer>();
        if (!m_scene)
        {
            Log::Warning("Warning: path tracer '%s' expected an initialised scene container but none was found.");
        }*/

        return true;
    }

    __host__ bool Host::NNanoSDF::IsClickablePoint(const UIViewCtx& viewCtx) const
    {
        return GetWorldSpaceBoundingBox().Contains(viewCtx.mousePos);
    }

    __host__ BBox2f Host::NNanoSDF::ComputeObjectSpaceBoundingBox()
    {
        return m_params.viewport.objectBounds;
    }

    __host__ bool Host::NNanoSDF::Serialise(Json::Node& node, const int flags) const
    {
        DrawableObject::Serialise(node, flags);

        Json::Node lookNode = node.AddChildObject("viewport");
        lookNode.AddVector("dims", m_params.viewport.dims);

        return true;
    }

    __host__ bool Host::NNanoSDF::Deserialise(const Json::Node& node, const int flags)
    {
        bool isDirty = DrawableObject::Deserialise(node, flags);

        Json::Node viewportNode = node.GetChildObject("viewport", flags);
        if (viewportNode)
        {
            isDirty |= viewportNode.GetVector("dims", m_params.viewport.dims, flags);
        }

        if (isDirty)
        {
            SignalDirty({ kDirtyParams });
        }

        return isDirty;
    }

    __host__ bool Host::NNanoSDF::OnDelegateAction(const std::string& stateID, const VirtualKeyMap& keyMap, const UIViewCtx& viewCtx)
    {
        const auto& bBox = GetWorldSpaceBoundingBox();
        const vec2 mouseNorm = (viewCtx.mousePos - bBox.Centroid()) / bBox.Dimensions();


        return true;
    }
}