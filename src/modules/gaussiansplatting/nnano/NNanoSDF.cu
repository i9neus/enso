#define CUDA_DEVICE_GLOBAL_ASSERTS

#include "NNanoSDF.cuh"
#include "core/math/ColourUtils.cuh"
#include "core/math/Hash.cuh"
#include "core/assets/AssetContainer.cuh"
#include "core/containers/Vector.cuh"
#include "core/assets/GenericObjectContainer.cuh"

#include "io/json/JsonUtils.h"
//#include "core/AccumulationBuffer.cuh"

namespace Enso
{
    __host__ __device__ NNanoSDFParams::NNanoSDFParams()
    {
        viewport.dims = ivec2(0);
        hasValidScene = false;
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

    __device__ void Device::NNanoSDF::Render()
    {
        const ivec2 xyViewport = kKernelPos<ivec2>();
        if (xyViewport.x < 0 || xyViewport.x >= m_params.viewport.dims.x || xyViewport.y < 0 || xyViewport.y >= m_params.viewport.dims.y) { return; }

        
    }
    DEFINE_KERNEL_PASSTHROUGH(Render);

    __device__ void Device::NNanoSDF::Composite(Device::ImageRGBA* deviceOutputImage)
    {
        CudaAssertDebug(deviceOutputImage);

        // TODO: Make alpha compositing a generic operation inside the Image class.
        const ivec2 xyAccum = kKernelPos<ivec2>();
        const ivec2 xyViewport = xyAccum + deviceOutputImage->Dimensions() / 2 - m_objects.meanAccumBuffer->Dimensions() / 2;

        /*BBox2i border(0, 0, m_params.viewport.dims.x, m_params.viewport.dims.y);
        if(border.PointOnPerimiter(xyAccum, 2))
        {
            deviceOutputImage->At(xyViewport) = vec4(1.0f);
        }*/
        if (xyAccum.x < m_params.viewport.dims.x && xyAccum.y < m_params.viewport.dims.y)
        {
            //if (xyAccum.x < m_params.viewport.dims.x / 2)
            {
                //const vec4& varL = m_objects.varAccumBuffer->At(xyAccum);
                const vec4& texel = m_objects.meanAccumBuffer->At(xyAccum);
                vec3 L = texel.xyz / fmaxf(1.f, texel.w);
                L = pow(L, 0.7f);

                deviceOutputImage->At(xyViewport) = vec4(L, 1.0f);

                //deviceOutputImage->At(xyViewport) = vec4(varL.xyz / fmaxf(1.f, varL.w) - sqr(meanL.xyz / fmaxf(1.f, meanL.w)), 1.f);
                //deviceOutputImage->At(xyViewport) = vec4(varL.xyz / sqr(fmaxf(1.f, varL.w)), 1.f);
            }
            /*else
            {
                const vec3& denoisedL = m_objects.denoisedBuffer->At(xyAccum);
                deviceOutputImage->At(xyViewport) = vec4(denoisedL, 1.f);
            }*/
        }
    }
    DEFINE_KERNEL_PASSTHROUGH_ARGS(Composite);

    __host__ __device__ bool Device::NNanoSDF::IsClickablePoint(const UIViewCtx& viewCtx) const
    {
        return GetWorldBBox().Contains(viewCtx.mousePos);
    }

    __host__ __device__ vec4 Device::NNanoSDF::EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx, const bool isMouseTest) const
    {
        if (!GetWorldBBox().Contains(pWorld)) { return vec4(0.0f); }

#ifdef __CUDA_ARCH__
        const vec2 pObject = ToObjectSpace(pWorld);
        const ivec2 pPixel = ivec2(vec2(m_params.viewport.dims) * (pObject - m_params.viewport.objectBounds.lower) / m_params.viewport.objectBounds.Dimensions());

        if (!m_params.hasValidScene)
        {
            const float hatch = step(0.8f, fract(0.05f * dot(pWorld / viewCtx.dPdXY, vec2(1.f))));
            return vec4(kOne * hatch * 0.1f, 1.f);
        }
        else if (pPixel.x >= 0 && pPixel.x < m_params.viewport.dims.x && pPixel.y >= 0 && pPixel.y < m_params.viewport.dims.y)
        {
            //if (pPixel.x < m_params.viewport.dims.x / 2)
            {
                const vec4& texel = m_objects.meanAccumBuffer->At(pPixel);
                vec3 L = texel.xyz / fmaxf(1.f, texel.w);
                L = pow(L, 0.7f);

                return vec4(L, 1.0f);

                //deviceOutputImage->At(xyViewport) = vec4(varL.xyz / fmaxf(1.f, varL.w) - sqr(meanL.xyz / fmaxf(1.f, meanL.w)), 1.f);
                //deviceOutputImage->At(xyViewport) = vec4(varL.xyz / sqr(fmaxf(1.f, varL.w)), 1.f);
            }
            /*else
            {
                const vec3& denoisedL = m_objects.denoisedBuffer->At(pPixel);
                return vec4(denoisedL, 1.f);
            }*/
        }
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
        RenderableObject::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::RenderableObject>(cu_deviceInstance));

        constexpr int kViewportWidth = 1200;
        constexpr int kViewportHeight = 675;

        // Create some Cuda objects
        m_hostAccumBuffer = AssetAllocator::CreateChildAsset<Host::ImageRGBW>(*this, "meanAccumBufferMean", kViewportWidth, kViewportHeight, nullptr);

        m_deviceObjects.meanAccumBuffer = m_hostAccumBuffer->GetDeviceInstance();

        const vec2 boundHalf = 0.25 * ((kViewportHeight > kViewportWidth) ?
            vec2(1.f, float(kViewportHeight) / float(kViewportWidth)) :
            vec2(float(kViewportWidth) / float(kViewportHeight), 1.f));

        m_params.viewport.dims = ivec2(kViewportWidth, kViewportHeight);
        m_params.viewport.objectBounds = BBox2f(-boundHalf, boundHalf);

        Cascade({ kDirtySceneObjectChanged });
    }

    __host__ Host::NNanoSDF::~NNanoSDF() noexcept
    {
        m_hostAccumBuffer.DestroyAsset();

        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);
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

    __host__ void Host::NNanoSDF::Bind(GenericObjectContainer& objects)
    {


    }

    __host__ void Host::NNanoSDF::Render()
    {
        if (IsDirty(kDirtySceneObjectChanged))
        {
            m_hostAccumBuffer->Clear(vec4(0.f));
            SignalDirty(kDirtyViewportRedraw);
        }

        //KernelPrepare << <1, 1 >> > (cu_deviceInstance, m_dirtyFlags);

        //if (RenderableObject::m_params.frameIdx > 10) return;

        dim3 blockSize, gridSize;
        KernelParamsFromImage(m_hostAccumBuffer, blockSize, gridSize);

        // Accumulate the frame
        KernelRender << < gridSize, blockSize, 0, m_hostStream >> > (cu_deviceInstance);

        // Denoise if necessary
        /*if (m_params.frameIdx % 500 == 0)
        {
            KernelDenoise << < gridSize, blockSize, 0, m_hostStream >> > (cu_deviceInstance);
        }*/

        IsOk(cudaDeviceSynchronize());

        // If there's no user interaction, signal the viewport to update intermittently to save compute
        constexpr float kViewportUpdateInterval = 1. / 2.f;
        if (m_redrawTimer.Get() > kViewportUpdateInterval)
        {
            SignalDirty(kDirtyViewportRedraw);
            m_redrawTimer.Reset();
        }

        if (m_renderTimer.Get() > 1.)
        {
            //Log::Debug("Frame: %i", RenderableObject::m_params.frameIdx);
            m_renderTimer.Reset();
        }
    }

    __host__ void Host::NNanoSDF::Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const
    {
        dim3 blockSize, gridSize;
        KernelParamsFromImage(m_hostAccumBuffer, blockSize, gridSize);

        KernelComposite << < gridSize, blockSize, 0, m_hostStream >> > (cu_deviceInstance, hostOutputImage->GetDeviceInstance());
        IsOk(cudaDeviceSynchronize());
    }

    __host__ void Host::NNanoSDF::Clear()
    {
        m_hostAccumBuffer->Clear(vec4(0.f));

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