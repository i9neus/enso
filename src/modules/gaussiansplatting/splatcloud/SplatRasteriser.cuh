#pragma once

#include "core/2d/Ctx.cuh"
#include "core/2d/DrawableObject.cuh"
#include "core/2d/RenderableObject.cuh"

#include "core/assets/DirtinessFlags.cuh"
#include "core/containers/Image.cuh"
#include "core/assets/GenericObject.cuh"
#include "core/utils/HighResolutionTimer.h"

#include "../FwdDecl.cuh"
//#include "core/3d/BidirectionalTransform.cuh"

namespace Enso
{         
    namespace Device { class Tracable; }
    struct Ray;
    struct HitCtx;
    
    struct SplatRasteriserParams
    {
        __device__ void Validate() const {}

        struct
        {
            ivec2 dims;
            BBox2f objectBounds;
        }
        viewport;

        bool hasValidSplatCloud = false;
    };

    struct SplatRasteriserObjects
    {
        __host__ __device__ void Validate() const
        {
            CudaAssert(frameBuffer);      
        }
        
        Device::ImageRGBW*                  frameBuffer = nullptr;
        Device::Vector<GaussianPoint>*      pointCloud = nullptr;
        Device::Camera*                     activeCamera = nullptr;
    };

    namespace Host { class SplatRasteriser; }

    namespace Device
    {
        class SplatRasteriser : public Device::DrawableObject, public Device::RenderableObject
        {
            friend Host::SplatRasteriser;

        public:
            __host__ __device__ SplatRasteriser() {}
            
            //__device__ void Prepare(const uint dirtyFlags);
            __device__ void Render();

            __host__ __device__ void Synchronise(const SplatRasteriserParams& params);
            __device__ void Synchronise(const SplatRasteriserObjects& objects);

            __host__ __device__ virtual vec4    EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx, const bool isMouseTest) const override final;

        private:
            __device__ int Trace(Ray& ray, HitCtx& hit) const;

            SplatRasteriserParams            m_params;
            SplatRasteriserObjects           m_objects;

            //Device::ComponentContainer        m_scene;
        };
    }

    namespace Host
    {
        class SplatRasteriser : public Host::DrawableObject, public Host::RenderableObject
        {
        public:
            __host__                    SplatRasteriser(const Asset::InitCtx& initCtx, const AssetHandle<const Host::GenericObjectContainer>& genericObjects);
            __host__ virtual            ~SplatRasteriser() noexcept;

            __host__ virtual void       Render() override final;
            __host__ void               Clear();

            __host__ static AssetHandle<Host::GenericObject> Instantiate(const std::string& id, const Host::Asset& parentAsset, const AssetHandle<const Host::GenericObjectContainer>& genericObjects);
            __host__ static const std::string  GetAssetClassStatic() { return "splatrasteriser"; }
            __host__ virtual std::string       GetAssetClass() const override final { return GetAssetClassStatic(); }

            __host__ virtual void       Bind(GenericObjectContainer& objects) override final;
            __host__ virtual bool       Serialise(Json::Node& rootNode, const int flags) const override final;
            __host__ virtual bool       Deserialise(const Json::Node& rootNode, const int flags) override final;
            __host__ virtual BBox2f     ComputeObjectSpaceBoundingBox() override final;
            __host__ virtual bool       HasOverlay() const override { return true; }
            __host__ virtual bool       IsClickablePoint(const UIViewCtx& viewCtx) const override final;
            __host__ virtual bool       IsDelegatable() const override final { return true; }

            __host__ virtual Device::SplatRasteriser* GetDeviceInstance() const override final
            {
                return cu_deviceInstance;
            }

        protected:
            __host__ virtual void       OnSynchroniseDrawableObject(const uint syncFlags) override final;
            __host__ virtual bool       OnCreateDrawableObject(const std::string& stateID, const UIViewCtx& viewCtx, const vec2& mousePosObject) override final;
            __host__ virtual bool       OnRebuildDrawableObject() override final;

        private:
            __host__ void               RebuildSplatCloud();

            template<typename T>
            __host__ void           KernelParamsFromImage(const AssetHandle<Host::Image<T>>& image, dim3& blockSize, dim3& gridSize) const
            {
                const auto& meta = image->GetMetadata();
                blockSize = dim3(16, 16, 1);
                gridSize = dim3((meta.Width() + 15) / 16, (meta.Height() + 15) / 16, 1);
            }

            Device::SplatRasteriser*          cu_deviceInstance = nullptr;
            Device::SplatRasteriser           m_hostInstance;
            SplatRasteriserObjects            m_objects;
            SplatRasteriserParams             m_params;
            HighResolutionTimer               m_wallTime;
            HighResolutionTimer               m_renderTimer;
            HighResolutionTimer               m_redrawTimer;

            AssetHandle<Host::ImageRGBW>      m_hostFrameBuffer;

            AssetHandle<Host::SceneContainer> m_hostSceneContainer;
            AssetHandle<Host::Camera>         m_hostActiveCamera;
            AssetHandle<Host::GaussianPointCloud> m_gaussianPointCloud;

            AssetHandle<Host::Vector<BidirectionalTransform>> m_hostTransforms;
        };
    }
}