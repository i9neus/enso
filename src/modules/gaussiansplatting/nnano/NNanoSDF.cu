#define CUDA_DEVICE_GLOBAL_ASSERTS

#include "NNanoSDF.cuh"
#include "NNanoEvaluator.cuh"
#include "SDFQuadraticSpline.cuh"

#include "core/math/ColourUtils.cuh"
#include "core/math/Hash.cuh"
#include "core/math/MathUtils.cuh"
#include "core/assets/AssetContainer.cuh"
#include "core/containers/Vector.cuh"
#include "core/assets/GenericObjectContainer.cuh"
#include "core/2d/Transform2D.cuh"
#include "core/2d/bih/BIH.cuh"

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
        printf("VALIDATING... %i x %i\n", viewport.dims.x, viewport.dims.y);
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
        if (!GetWorldBBox().Contains(pWorld)) { return vec4(0.0f); }

        if (isMouseTest) { return vec4(0.); } // Mouse testing disabled for now

        const vec2 pObject = ToObjectSpace(pWorld);
        const vec2 pScreen = (pObject - m_params.viewport.objectBounds.lower) / m_params.viewport.objectBounds.Dimensions();
        vec4 L(0.0f);

#ifdef __CUDA_ARCH__

        // Sample the evaluation buffer
        const ivec2 pPixel = ivec2(vec2(m_objects.evalBuffer->Dimensions()) * pScreen);
        if (pPixel.x >= 0 && pPixel.x < m_objects.evalBuffer->Width() && pPixel.y >= 0 && pPixel.y < m_objects.evalBuffer->Height())
        {
            L = vec4(*reinterpret_cast<const vec3*>(m_objects.evalBuffer->At(pPixel)), 1);         
        }
#endif

        // Evaluate the SDF
        /*const vec2 pView = TransformNormalisedScreenToView(pScreen, m_params.viewport.dims);
        if (m_objects.sdf)
        {
            const vec3 f = m_objects.sdf->Evaluate(pView);
            L = Blend(L, vec4(kOne, cosf(10. * kTwoPi * f.x) * 0.5 + 0.5));
        }*/

        return L;
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

        // Create some objects
        m_hostSDF = AssetAllocator::CreateChildAsset<Host::SDFQuadraticSpline>(*this, "spline");
        m_hostEvalBuffer = AssetAllocator::CreateChildAsset<Host::DualImage3f>(*this, "evalBuffer", 1200, 675, nullptr);

        m_deviceObjects.sdf = m_hostSDF->GetDeviceInstance(); 
        m_deviceObjects.evalBuffer = m_hostEvalBuffer->GetDeviceInstance();

        m_evaluator = AssetAllocator::CreateChildAsset<NNanoEvaluator>(*this, "evaluator", m_hostSDF, m_params.viewport.objectBounds, ivec2(1200, 675), m_hostEvalBuffer);
        
        Synchronise(kSyncObjects | kSyncParams);
        Cascade({ kDirtySceneObjectChanged });
    }

    __host__ Host::NNanoSDF::~NNanoSDF() noexcept
    {
        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance); 

        m_hostSDF.DestroyAsset();
        m_hostEvalBuffer.DestroyAsset();
    }

    __host__ void Host::NNanoSDF::OnSynchroniseDrawableObject(const uint syncFlags)
    {
        // Only sync the objects if a SceneContainer has been bound
        if (syncFlags & kSyncObjects)
        {
            SynchroniseObjects<Device::NNanoSDF>(cu_deviceInstance, m_deviceObjects);
            
            // Synchronise the host copy so we can query it
            m_hostInstance.m_objects.sdf = &m_hostSDF->GetHostInstance();
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