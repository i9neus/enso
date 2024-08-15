#pragma once

#include "core/2d/Ctx.cuh"
#include "core/2d/DrawableObject.cuh"
#include "core/2d/RenderableObject.cuh"

#include "core/assets/DirtinessFlags.cuh"
#include "core/containers/Image.cuh"
#include "core/assets/GenericObject.cuh"

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
        Device::Vector<Device::Tracable*>*  tracables = nullptr;
        Device::Vector<Device::Light*>*     lights = nullptr;
        Device::Vector<Device::Texture2D*>* textures = nullptr;
        Device::Vector<Device::Material*>*  materials = nullptr;
    };

    namespace Host { class PathTracer; }

    namespace Device
    {
        class PathTracer : public Device::DrawableObject, public Device::RenderableObject
        {
            friend Host::PathTracer;

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
            __device__ int Trace(Ray& ray, HitCtx& hit) const;

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