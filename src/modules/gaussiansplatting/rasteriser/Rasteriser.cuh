#pragma once

#include "core/2d/Ctx.cuh"
#include "core/2d/DrawableObject.cuh"
#include "core/2d/RenderableObject.cuh"

#include "core/DirtinessFlags.cuh"
#include "core/Image.cuh"
#include "core/GenericObject.cuh"
#include "core/HighResolutionTimer.h"

#include "../FwdDecl.cuh"
//#include "core/3d/BidirectionalTransform.cuh"

namespace Enso
{         
    namespace Device { class Tracable; }
    struct Ray;
    struct HitCtx;
    
    struct RasteriserParams
    {
        __host__ __device__ RasteriserParams();
        __device__ void Validate() const;

        struct
        {
            ivec2 dims;
            BBox2f objectBounds;
        }
        viewport;

        int frameIdx;
        float wallTime;
        bool hasValidScene;
    };

    struct RasteriserObjects
    {
        __host__ __device__ void Validate() const
        {
            CudaAssert(frameBuffer);      
        }
        
        Device::ImageRGBW*                  frameBuffer = nullptr;
        Device::Camera*                     activeCamera = nullptr;
        Device::Vector<Device::Tracable*>*  tracables = nullptr;
    };

    namespace Host { class Rasteriser; }

    namespace Device
    {
        class Rasteriser : public Device::DrawableObject
        {
            friend Host::Rasteriser;

        public:
            __host__ __device__ Rasteriser() {}
            
            //__device__ void Prepare(const uint dirtyFlags);
            __device__ void Render();

            __host__ __device__ void Synchronise(const RasteriserParams& params);
            __device__ void Synchronise(const RasteriserObjects& objects);

            __host__ __device__ uint            OnMouseClick(const UIViewCtx& viewCtx) const;
            __host__ __device__ virtual vec4    EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx, const bool isMouseTest) const override final;

        private:
            __device__ int Trace(Ray& ray, HitCtx& hit) const;

            RasteriserParams            m_params;
            RasteriserObjects           m_objects;

            //Device::ComponentContainer        m_scene;
        };
    }

    namespace Host
    {
        class Rasteriser : public Host::DrawableObject, Host::RenderableObject
        {
        public:
            __host__                    Rasteriser(const Asset::InitCtx& initCtx, const AssetHandle<const Host::GenericObjectContainer>& genericObjects);
            __host__ virtual            ~Rasteriser() noexcept;

            __host__ virtual void       Render() override final;
            __host__ void               Clear();

            __host__ static AssetHandle<Host::GenericObject> Instantiate(const std::string& id, const Host::Asset& parentAsset, const AssetHandle<const Host::GenericObjectContainer>& genericObjects);
            __host__ static const std::string  GetAssetClassStatic() { return "rasteriser"; }
            __host__ virtual std::string       GetAssetClass() const override final { return GetAssetClassStatic(); }

            __host__ virtual void       Bind(GenericObjectContainer& objects) override final;
            __host__ virtual uint       OnMouseClick(const UIViewCtx& viewCtx) const override final;
            __host__ virtual bool       Serialise(Json::Node& rootNode, const int flags) const override final;
            __host__ virtual bool       Deserialise(const Json::Node& rootNode, const int flags) override final;
            __host__ virtual BBox2f     ComputeObjectSpaceBoundingBox() override final;
            __host__ virtual bool       HasOverlay() const override { return true; }

            __host__ virtual Device::Rasteriser* GetDeviceInstance() const override final
            {
                return cu_deviceInstance;
            }

        protected:
            __host__ virtual void       OnSynchroniseDrawableObject(const uint syncFlags) override final;
            __host__ virtual bool       OnCreateDrawableObject(const std::string& stateID, const UIViewCtx& viewCtx, const vec2& mousePosObject) override final;
            __host__ virtual bool       OnRebuildDrawableObject() override final;

        private:
            template<typename T>
            __host__ void           KernelParamsFromImage(const AssetHandle<Host::Image<T>>& image, dim3& blockSize, dim3& gridSize) const
            {
                const auto& meta = image->GetMetadata();
                blockSize = dim3(16, 16, 1);
                gridSize = dim3((meta.Width() + 15) / 16, (meta.Height() + 15) / 16, 1);
            }

            Device::Rasteriser*               cu_deviceInstance = nullptr;
            Device::Rasteriser                m_hostInstance;
            RasteriserObjects                 m_deviceObjects;
            RasteriserParams                  m_params;
            HighResolutionTimer               m_wallTime;
            HighResolutionTimer               m_renderTimer;
            HighResolutionTimer               m_redrawTimer;

            AssetHandle<Host::ImageRGBW>      m_hostFrameBuffer;

            AssetHandle<Host::SceneContainer> m_hostSceneContainer;
            AssetHandle<Host::Camera>         m_hostActiveCamera;

            AssetHandle<Host::Vector<BidirectionalTransform>> m_hostTransforms;
        };
    }
}