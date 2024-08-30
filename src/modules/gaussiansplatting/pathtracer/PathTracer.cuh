#pragma once

#include "core/2d/Ctx.cuh"
#include "core/2d/DrawableObject.cuh"
#include "core/2d/RenderableObject.cuh"
#include "core/3d/Ctx.cuh"
#include "core/3d/Ray.cuh"

#include "core/assets/DirtinessFlags.cuh"
#include "core/containers/Image.cuh"
#include "core/assets/GenericObject.cuh"
#include "../scene/SceneContainer.cuh"

#include "../FwdDecl.cuh"
#include "NLM.cuh"
//#include "core/3d/BidirectionalTransform.cuh"

namespace Enso
{         
    namespace Device { class Tracable; }
    struct Ray;
    struct HitCtx;
    
    struct PathTracerParams
    {
        __host__ __device__ PathTracerParams();
        __device__ void Validate() const;

        struct
        {
            ivec2 dims;
            BBox2f objectBounds;
        } 
        viewport;
        
        bool hasValidScene;
    };

    struct PathTracerObjects
    {
        __host__ __device__ void Validate() const
        {
            CudaAssert(transforms);
            CudaAssert(meanAccumBuffer);
            CudaAssert(varAccumBuffer);
            CudaAssert(denoisedBuffer);           
        }
        
        Device::ImageRGBW*                  meanAccumBuffer = nullptr;
        Device::ImageRGBW*                  varAccumBuffer = nullptr;
        Device::ImageRGB*                   denoisedBuffer = nullptr;
        Device::Vector<BidirectionalTransform>* transforms = nullptr;

        Device::Camera*                     activeCamera = nullptr;

        Device::SceneContainer              scene;
    };

    namespace Host { class PathTracer; }

    namespace Device
    {
        class PathTracer : public Device::DrawableObject, public Device::RenderableObject
        {
            friend Host::PathTracer;

            class RayStack
            {
            private:
                Ray m_stack[2];
                int m_top;

            public:
                __device__ RayStack() : m_top(-1) {}
                __device__ __forceinline__ Ray& Push() 
                { 
                    CudaAssertDebug(m_top < 1, "Extant ray stack overflow");
                    return m_stack[++m_top]; 
                }
                __device__ __forceinline__ void Push(const Ray& ray) 
                { 
                    CudaAssertDebug(m_top < 1, "Extant ray stack overflow");
                    m_stack[++m_top] = ray; 
                }
                __device__ __forceinline__ Ray& Pop() { return m_stack[m_top--]; }
                __device__ __forceinline__ Ray& Top() { return m_stack[m_top]; }

                __device__ __forceinline__ bool IsEmpty() const { return m_top < 0; }
                __device__ __forceinline__ operator bool() const { return m_top >= 0; }
            };

        public:
            __host__ __device__ PathTracer() {}
            
            //__device__ void Prepare(const uint dirtyFlags);
            __device__ void Render();
            __device__ void Denoise();
            __device__ void Composite(Device::ImageRGBA* outputImage, const bool isValidScene);

            __host__ __device__ void Synchronise(const PathTracerParams& params);
            __device__ void Synchronise(const PathTracerObjects& objects);

            __host__ __device__ bool            IsClickablePoint(const UIViewCtx& viewCtx) const;
            __host__ __device__ virtual vec4    EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx, const bool isMouseTest) const override final;

        private:
            __device__ const Device::Tracable*  Trace(Ray& ray, HitCtx& hit) const;
            __device__ float                    SampleEmitter(const Ray& incident, RayStack& extantStack, const HitCtx& hit, const Material& material, const LightSampler* lightSampler, const vec2& xi) const;
            __device__ float                    SampleBxDF(const Ray& incident, RayStack& extantStack, const HitCtx& hit, const Material& material, const LightSampler* lightSampler, const vec2& xi, const bool isDirectSample) const;
            __device__ void                     Shade(const Ray& incidentRay, RayStack& extantStack, HitCtx& hit, RenderCtx& renderCtx, const Material& material, int renderMode, vec3& L) const;

        private:
            PathTracerParams            m_params;
            PathTracerObjects           m_objects;

            NLMDenoiser                 m_nlm;

            //Device::ComponentContainer        m_scene;
        };
    }

    namespace Host
    {
        class PathTracer : public Host::DrawableObject, public Host::RenderableObject
        {
        public:
            __host__                    PathTracer(const Asset::InitCtx& initCtx, const AssetHandle<const Host::GenericObjectContainer>& genericObjects);
            __host__ virtual            ~PathTracer() noexcept;

            __host__ virtual void       Render() override final;
            __host__ void               Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const;
            __host__ void               Clear();

            __host__ static AssetHandle<Host::GenericObject> Instantiate(const std::string& id, const Host::Asset& parentAsset, const AssetHandle<const Host::GenericObjectContainer>& genericObjects);
            __host__ static const std::string  GetAssetClassStatic() { return "pathtracer"; }
            __host__ virtual std::string       GetAssetClass() const override final { return GetAssetClassStatic(); }

            __host__ virtual void       Bind(GenericObjectContainer& objects) override final;
            __host__ virtual bool       Serialise(Json::Node& rootNode, const int flags) const override final;
            __host__ virtual bool       Deserialise(const Json::Node& rootNode, const int flags) override final;
            __host__ virtual BBox2f     ComputeObjectSpaceBoundingBox() override final;
            __host__ virtual bool       HasOverlay() const override { return true; }
            __host__ virtual bool       IsClickablePoint(const UIViewCtx& viewCtx) const override final;
            __host__ virtual bool       IsDelegatable() const override final { return true; }
            __host__ virtual bool       OnDelegateAction(const std::string& stateID, const VirtualKeyMap& keyMap, const UIViewCtx& viewCtx) override final;

            __host__ virtual Device::PathTracer* GetDeviceInstance() const override final
            {
                return cu_deviceInstance;
            }

        protected:
            __host__ virtual void       OnSynchroniseDrawableObject(const uint syncFlags) override final;
            __host__ virtual bool       OnCreateDrawableObject(const std::string& stateID, const UIViewCtx& viewCtx, const vec2& mousePosObject) override final;
            __host__ virtual bool       OnRebuildDrawableObject() override final;

        private:
            __host__ void               CreateScene();

            template<typename T>
            __host__ void           KernelParamsFromImage(const AssetHandle<Host::Image<T>>& image, dim3& blockSize, dim3& gridSize) const
            {
                const auto& meta = image->GetMetadata();
                blockSize = dim3(16, 16, 1);
                gridSize = dim3((meta.Width() + 15) / 16, (meta.Height() + 15) / 16, 1);
            }

            Device::PathTracer*               cu_deviceInstance = nullptr;
            Device::PathTracer                m_hostInstance;
            PathTracerObjects                 m_deviceObjects;
            PathTracerParams                  m_params;
            HighResolutionTimer               m_renderTimer;
            HighResolutionTimer               m_redrawTimer;

            AssetHandle<Host::ImageRGBW>      m_hostMeanAccumBuffer;
            AssetHandle<Host::ImageRGBW>      m_hostVarAccumBuffer;
            AssetHandle<Host::ImageRGB>       m_hostDenoisedBuffer;

            AssetHandle<Host::SceneContainer> m_hostSceneContainer;
            AssetHandle<Host::Camera>         m_hostActiveCamera;

            AssetHandle<Host::Vector<BidirectionalTransform>> m_hostTransforms;
        };
    }
}