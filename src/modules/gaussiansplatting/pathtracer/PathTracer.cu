#define CUDA_DEVICE_GLOBAL_ASSERTS

#include "PathTracer.cuh"
#include "core/math/ColourUtils.cuh"
#include "core/math/Hash.cuh"
#include "core/assets/AssetContainer.cuh"
#include "core/3d/Transform.cuh"
#include "core/containers/Vector.cuh"
#include "core/3d/Cameras.cuh"
//#include "Scene.cuh"
//#include "Integrator.cuh"
#include "core/assets/GenericObjectContainer.cuh"
//#include "Geometry.cuh"
#include "core/math/samplers/MersenneTwister.cuh"
#include "core/math/samplers/Dither.cuh"

#include "../scene/SceneContainer.cuh"
#include "../scene/cameras/Camera.cuh"
#include "../scene/materials/Material.cuh"
#include "../scene/lights/LightSampler.cuh"
#include "../scene/lights/QuadLight.cuh"
#include "../scene/textures/Texture2D.cuh"
#include "../scene/tracables/Tracable.cuh"

#include "io/json/JsonUtils.h"
//#include "core/AccumulationBuffer.cuh"

namespace Enso
{        
    #define kMatInvalid -1

    #define kModePathTraced 0
    #define kModeNEE 1

    #define kGenerateNothing 0
    #define kGeneratedDirect 1
    #define kGeneratedIndirect 2
    
    __host__ __device__ PathTracerParams::PathTracerParams()
    {
        viewport.dims = ivec2(0);  
        hasValidScene = false;
    }

    __host__ __device__ void PathTracerParams::Validate() const
    {
        CudaAssert(viewport.dims.x != 0 && viewport.dims.y != 0);
    }   

    __host__ __device__ void Device::PathTracer::Synchronise(const PathTracerParams& params) 
    {
        m_params = params;   
        m_nlm.Initialise(10, 2, 2.f, 2.f);
     }
    
    __device__ void Device::PathTracer::Synchronise(const PathTracerObjects& objects) 
    { 
        objects.Validate(); 
        m_objects = objects; 
        m_nlm.Initialise(m_objects.meanAccumBuffer, m_objects.varAccumBuffer);
    }

    __device__ __forceinline__ float PowerHeuristic(float pdf1, float pdf2)
    {
        return saturatef(sqr(pdf1) / fmaxf(1e-10, sqr(pdf1) + sqr(pdf2)));
    }

    __device__ float Device::PathTracer::SampleEmitter(const Ray& incident, RayStack& extantStack, const HitCtx& hit, const Material& material, const LightSampler* lightSampler, const vec2& xi) const
    {
        auto& extant = extantStack.Push();
        float emitterPdf = lightSampler->Sample(incident, extant, hit, xi);
        if (emitterPdf <= 0.)
        {
            extantStack.Pop();
            return 0.;
        }
 
        //float bxdfPdf = material.Evaluate(-incident.od.d, extant.od.d, hit.n);
        vec3 materialWeight = kOne;
        float bxdfPdf = material.Evaluate(incident, extant, hit, materialWeight);

        // Lambert cosine factor
        bxdfPdf *= dot(extant.od.d, hit.n);

        // Apply power heuristic up-weighted by a factor to two to account for stochastic branching
        extant.weight *= 2. * bxdfPdf * materialWeight * PowerHeuristic(emitterPdf, bxdfPdf);

        return emitterPdf;
    }

    __device__ float Device::PathTracer::SampleBxDF(const Ray& incident, RayStack& extantStack, const HitCtx& hit, const Material& material, const LightSampler* lightSampler, const vec2& xi, const bool isDirectSample) const
    {
        vec3 o;
        vec3 kickoff = hit.n * 1e-4;
        
        // Sample the BxDF
        vec3 materialWeight = kOne;
        //float bxdfPdf = material.Sample(xi, -incident.od.d, hit.n, o, materialWeight);      
        float bxdfPdf = material.Sample(xi, incident, hit, o, materialWeight);

        // Create the ray
        auto& extant = extantStack.Push();
        extant.Construct(incident.Point(), o, kickoff, incident.weight * materialWeight, incident.depth + 1, incident.InheritedFlags());

        // If this isn't a perfect specular BxDF, flag the ray as scattered
        if (!material.IsPerfectSpecular()) { extant.flags |= kRayScattered; }

        // If this is a light samaple, compute the PDF of the emitter and apply the power heuristic
        if (isDirectSample)
        {
            CudaAssertDebug(lightSampler);
            const float emitterPdf = lightSampler->Evaluate(extant, hit);
            extant.weight *= 2. * PowerHeuristic(bxdfPdf, emitterPdf);
            extant.flags |= kRayDirectSampleBxDF;
        }

        return bxdfPdf;
    }

    __device__ void Device::PathTracer::Shade(const Ray& incidentRay, RayStack& extantStack, HitCtx& hit, RenderCtx& renderCtx, const Material& material, int renderMode, vec3& L) const
    {
        // Generate some random numbers
        //vec4 xi = Rand(renderCtx.rng);
        vec4 xi = renderCtx.Rand(1 + incidentRay.depth);
        float xiSplit = fract(OrderedDither(renderCtx.viewport.xy) + float(renderCtx.frameIdx) / 16.);

        // Perfect specular BxDFs don't need light sampling
        if (material.IsPerfectSpecular()) { renderMode = kModePathTraced; }

        int genFlags = 0;

        // Sample the BxDF for the indirect contribution
        SampleBxDF(incidentRay, extantStack, hit, material, nullptr, xi.zw, false);

        // If we're in next-event estimation mode, stochastically sample either the emitter or the BxDF
        // IMPORTANT: Direct rays must be the last on the stack to prevent it overflowing
        if (renderMode == kModeNEE)
        {
            const LightSampler* light = (*m_objects.scene.lightSamplers)[0];
            
            if (xiSplit < 0.5)
            {
                SampleBxDF(incidentRay, extantStack, hit, material, light, xi.xy, true);
            }
            else
            {
                SampleEmitter(incidentRay, extantStack, hit, material, light, xi.xy);
            }
        }
    }

    // Trace scene geometry and return a pointer to the closest hit object
    __device__ const Device::Tracable* Device::PathTracer::Trace(Ray& ray, HitCtx& hitCtx) const
    {
        const Tracable* hitTracable = nullptr;
        for (int idx = 0; idx < m_objects.scene.tracables->size(); ++idx)
        {
            const Tracable* testTracable = (*m_objects.scene.tracables)[idx];
            if (testTracable->IntersectRay(ray, hitCtx))
            {
                hitTracable = testTracable;
            }
        }

        return hitTracable;
    }

    __device__ void Device::PathTracer::Render()
    {        
        const ivec2 xyViewport = kKernelPos<ivec2>();
        if (xyViewport.x < 0 || xyViewport.x >= m_params.viewport.dims.x || xyViewport.y < 0 || xyViewport.y >= m_params.viewport.dims.y) { return; }        

        // Get pointers to the object transforms
        const auto& tracables = *m_objects.scene.tracables;
        const auto& textures = *m_objects.scene.textures;

        // Create a render context
        RenderCtx renderCtx;
        renderCtx.rng.Initialise(HashOf(RenderableObject::m_params.frameIdx, xyViewport.x, xyViewport.y));
        renderCtx.qrng.Initialise(0, HashOf(xyViewport.x, xyViewport.y) + RenderableObject::m_params.frameIdx);
        renderCtx.viewport.dims = m_params.viewport.dims;
        renderCtx.viewport.xy = xyViewport;
        renderCtx.frameIdx = RenderableObject::m_params.frameIdx;

        // Transform into normalised sceen space
        const vec4 xi = renderCtx.Rand(0);
        const vec2 uvView = PixelToNormalisedScreen(vec2(xyViewport) + xi.xy, vec2(m_params.viewport.dims));
        
        RayStack extantStack;
        m_objects.activeCamera->CreateRay(uvView, xi.zw, extantStack.Push());

        int genFlags = kGeneratedIndirect;
        HitCtx hitCtx;
        vec3 L = kZero;
        //int renderMode = (xyViewport.x < m_params.viewport.dims.x * 0.5f) ? kModePathTraced : kModeNEE;
        const int renderMode = kModePathTraced;

        constexpr int kMaxPathDepth = 5;
        constexpr int kMaxIterations = 10;
        for (int rayIdx = 0; rayIdx < kMaxIterations && !extantStack.IsEmpty(); ++rayIdx)
        {
            // Pop the stack
            Ray ray = extantStack.Pop();         

            // Trace the ray
            const auto* hitTracable = Trace(ray, hitCtx);
            if (!hitTracable)
            {
                // Only accumulate environment contribution on indirect samples
                if(!ray.IsDirectSample() && m_objects.scene.envTexture)
                {
                    //L += kOne * ray.weight * 0.2;
                    const float env = luminance(m_objects.scene.envTexture->Evaluate(DirToEquirect(ray.od.d)).xyz);
                    L += ray.weight * powf(clamp(env, 0.f, 10.0f), 1 / 1.8f);
                }
                continue;
            }

            // Evaluate a direct ray
            if (ray.IsDirectSample())
            {
                // If we're hit a light, 
                if (hitTracable->IsLight() && !ray.IsBackfacing())
                {
                    // If this sample is a light ray, all we need to know is whether or not it hit the light. 
                    // If it did, just accumulate the weight which contains the radiant energy from the light sample. 
                    L += ray.weight;
                }
            }
            else
            {
                // Emitters don't reflect light so we don't need to shade them. Simply accumulate the emitted radiance and we're done.
                if (hitTracable->IsLight())
                {
                    if (!ray.IsBackfacing())
                    {
                        L += hitTracable->GetRadiance() * ray.weight;
                    }
                }
                // If we're at depth, skip this evaluation
                else if (ray.depth < kMaxPathDepth) 
                {
                    const Material& material = *(*m_objects.scene.materials)[hitTracable->GetMaterialIdx()];
                    Shade(ray, extantStack, hitCtx, renderCtx, material, renderMode, L);
                }
            }
        }

        auto& meanL = m_objects.meanAccumBuffer->At(xyViewport);
        auto& varL = m_objects.varAccumBuffer->At(xyViewport);

        // Unstable running variance
        //varL += vec4(sqr(L), 1.);
        
        // Welford's online algorithm
        varL += vec4((L - meanL.xyz / fmaxf(1.f, meanL.w)) * 
                     (L - (L + meanL.xyz) / (1.f + meanL.w)), 1.0f);

        meanL += vec4(L, 1.);

    }
    DEFINE_KERNEL_PASSTHROUGH(Render);

    __device__ void Device::PathTracer::Denoise()
    {        
        const ivec2 xyViewport = kKernelPos<ivec2>();
        if (xyViewport.x < m_params.viewport.dims.x && xyViewport.y < m_params.viewport.dims.y)
        {
            m_objects.denoisedBuffer->At(xyViewport) = m_nlm.FilterPixel(xyViewport);
        }
    }
    DEFINE_KERNEL_PASSTHROUGH(Denoise);

    __device__ void Device::PathTracer::Composite(Device::ImageRGBA* deviceOutputImage, const bool isValidScene)
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

    __host__ __device__ bool Device::PathTracer::IsClickablePoint(const UIViewCtx& viewCtx) const
    {
        return GetWorldBBox().Contains(viewCtx.mousePos);
    }

    __host__ __device__ vec4 Device::PathTracer::EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx, const bool isMouseTest) const
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

    __host__ AssetHandle<Host::GenericObject> Host::PathTracer::Instantiate(const std::string& id, const Host::Asset& parentAsset, const AssetHandle<const Host::GenericObjectContainer>& genericObjects)
    {
        return AssetAllocator::CreateChildAsset<Host::PathTracer>(parentAsset, id, genericObjects);
    }

    __host__ Host::PathTracer::PathTracer(const Asset::InitCtx& initCtx, const AssetHandle<const Host::GenericObjectContainer>& genericObjects) :
        DrawableObject(initCtx, &m_hostInstance),
        cu_deviceInstance(AssetAllocator::InstantiateOnDevice<Device::PathTracer>(*this))
    {        
        DrawableObject::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::DrawableObject>(cu_deviceInstance));
        RenderableObject::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::RenderableObject>(cu_deviceInstance));

        constexpr int kViewportWidth = 1200;
        constexpr int kViewportHeight = 675;

        // Create some Cuda objects
        m_hostMeanAccumBuffer = AssetAllocator::CreateChildAsset<Host::ImageRGBW>(*this, "meanAccumBufferMean", kViewportWidth, kViewportHeight, nullptr);
        m_hostVarAccumBuffer = AssetAllocator::CreateChildAsset<Host::ImageRGBW>(*this, "meanAccumBufferVar", kViewportWidth, kViewportHeight, nullptr);
        m_hostDenoisedBuffer = AssetAllocator::CreateChildAsset<Host::ImageRGB>(*this, "denoisedBuffer", kViewportWidth, kViewportHeight, nullptr);
        m_hostTransforms = AssetAllocator::CreateChildAsset<Host::Vector<BidirectionalTransform>>(*this, "transforms");

        m_deviceObjects.meanAccumBuffer = m_hostMeanAccumBuffer->GetDeviceInstance();
        m_deviceObjects.varAccumBuffer = m_hostVarAccumBuffer->GetDeviceInstance();
        m_deviceObjects.denoisedBuffer = m_hostDenoisedBuffer->GetDeviceInstance();
        m_deviceObjects.transforms = m_hostTransforms->GetDeviceInstance();

        const vec2 boundHalf = 0.25 * ((kViewportHeight > kViewportWidth) ?
            vec2(1.f, float(kViewportHeight) / float(kViewportWidth)) :
            vec2(float(kViewportWidth) / float(kViewportHeight), 1.f));

        m_params.viewport.dims = ivec2(kViewportWidth, kViewportHeight);
        m_params.viewport.objectBounds = BBox2f(-boundHalf, boundHalf);

        Cascade({ kDirtySceneObjectChanged });        
    }

    __host__ Host::PathTracer::~PathTracer() noexcept
    {
        m_hostMeanAccumBuffer.DestroyAsset();
        m_hostVarAccumBuffer.DestroyAsset();
        m_hostDenoisedBuffer.DestroyAsset();
        m_hostTransforms.DestroyAsset();

        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);
    }

    __host__ void Host::PathTracer::OnSynchroniseDrawableObject(const uint syncFlags)
    {
        // Only sync the objects if a SceneContainer has been bound
        if (syncFlags & kSyncObjects)
        {
            SynchroniseObjects<Device::PathTracer>(cu_deviceInstance, m_deviceObjects);
        }
        if (syncFlags & kSyncParams)
        {
            SynchroniseObjects<Device::PathTracer>(cu_deviceInstance, m_params);
            m_hostInstance.Synchronise(m_params);
        }
    }

    __host__ void Host::PathTracer::Bind(GenericObjectContainer& objects)
    {
        m_hostSceneContainer = objects.FindFirstOfType<Host::SceneContainer>();
        if (!m_hostSceneContainer)
        {
            Log::Warning("Warning! Path tracer '%s' could not bind to a valid SceneContainer object.", GetAssetID());
            m_params.hasValidScene = false;
        }
        else
        {
            // Copy the structure containing the scene object pointers 
            m_deviceObjects.scene = m_hostSceneContainer->GetDeviceObjects();

            if (m_hostSceneContainer->Cameras().empty())
            {
                Log::Warning("Warning! Path tracer '%s' found no cameras in the scene.");
                m_hostActiveCamera = nullptr;
            }
            else
            {
                m_hostActiveCamera = m_hostSceneContainer->Cameras().back();
                m_deviceObjects.activeCamera = m_hostActiveCamera->GetDeviceInstance();
            }

            m_params.hasValidScene = true;
        }

        Synchronise(kSyncParams | kSyncObjects);
    }

    __host__ void Host::PathTracer::Render()
    {
        if (!m_hostSceneContainer || !m_hostActiveCamera) { return; }

        //if (!IsClean()) { Synchronise(kSyncParams); }

        if (IsDirty(kDirtySceneObjectChanged))
        {
            m_hostMeanAccumBuffer->Clear(vec4(0.f));
            SignalDirty(kDirtyViewportRedraw);
        }

        //KernelPrepare << <1, 1 >> > (cu_deviceInstance, m_dirtyFlags);

        //if (RenderableObject::m_params.frameIdx > 10) return;

        dim3 blockSize, gridSize;
        KernelParamsFromImage(m_hostMeanAccumBuffer, blockSize, gridSize);

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

    __host__ void Host::PathTracer::Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const
    {
        dim3 blockSize, gridSize;
        KernelParamsFromImage(m_hostMeanAccumBuffer, blockSize, gridSize);

        KernelComposite << < gridSize, blockSize, 0, m_hostStream >> > (cu_deviceInstance, hostOutputImage->GetDeviceInstance(), m_hostSceneContainer != nullptr);
        IsOk(cudaDeviceSynchronize());
    }

    __host__ void Host::PathTracer::Clear()
    {
        m_hostMeanAccumBuffer->Clear(vec4(0.f));

        Synchronise(kSyncParams);
    }

    __host__ bool Host::PathTracer::OnCreateDrawableObject(const std::string& stateID, const UIViewCtx& viewCtx, const vec2& mousePosObject)
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

    __host__ bool Host::PathTracer::OnRebuildDrawableObject()
    {
        /*m_scene = m_componentContainer->GenericObjects().FindFirstOfType<Host::SceneContainer>();
        if (!m_scene)
        {
            Log::Warning("Warning: path tracer '%s' expected an initialised scene container but none was found.");
        }*/

        return true;
    }

    __host__ bool Host::PathTracer::IsClickablePoint(const UIViewCtx& viewCtx) const
    {
        return GetWorldSpaceBoundingBox().Contains(viewCtx.mousePos);
    }

    __host__ BBox2f Host::PathTracer::ComputeObjectSpaceBoundingBox()
    {
        return m_params.viewport.objectBounds;
    }

    __host__ bool Host::PathTracer::Serialise(Json::Node& node, const int flags) const
    {
        DrawableObject::Serialise(node, flags);

        Json::Node lookNode = node.AddChildObject("viewport");
        lookNode.AddVector("dims", m_params.viewport.dims);

        return true;
    }

    __host__ bool Host::PathTracer::Deserialise(const Json::Node& node, const int flags)
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

    __host__ bool Host::PathTracer::OnDelegateAction(const std::string& stateID, const VirtualKeyMap& keyMap, const UIViewCtx& viewCtx)
    {        
        const auto& bBox = GetWorldSpaceBoundingBox();
        const vec2 mouseNorm = (viewCtx.mousePos - bBox.Centroid()) / bBox.Dimensions();

        const float cameraPhi = kTwoPi * mix(-1., 1., mouseNorm.x);
        const float cameraDist = mix(0.5, 3., mouseNorm.y);
        const vec3 cameraPos = vec3(cos(cameraPhi), 0.5, sin(cameraPhi)) * cameraDist;

        Log::Debug("%s", cameraPos.format());
        
        m_hostActiveCamera->SetPosition(cameraPos);
        
        return true;
    }
}