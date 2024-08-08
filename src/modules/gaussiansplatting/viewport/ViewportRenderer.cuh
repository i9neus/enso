#pragma once

#include "core/GenericObject.cuh"
#include "core/2d/Ctx.cuh"
#include "../FwdDecl.cuh"
#include "core/Image.cuh"

namespace Enso
{         
    namespace Device { class Tracable; }
    
    struct ViewportRendererParams
    {
        __host__ __device__ ViewportRendererParams();
        __device__ void Validate() const {}

        UIGridCtx           gridCtx;
        UIViewCtx           viewCtx;
        
        struct
        {
            BBox2f                  mouseBBox;
            BBox2f                  lassoBBox;
            BBox2f                  selectedBBox;
            int                     numSelected;
            bool                    isLassoing = false;
        } 
        selectionCtx;

        struct
        {
            int                     viewport;
            int                     global;
        }
        frameIdx;
    };

    struct ViewportObjects
    {
        __device__ void Validate() const
        {
            CudaAssert(viewportObjects);
            CudaAssert(viewportBIH);
            CudaAssert(accumBuffer);
        }

        const Device::Vector<Device::DrawableObject*>*  viewportObjects = nullptr;
        const BIH2D<BIH2DFullNode>*                     viewportBIH = nullptr;
        
        Device::ImageRGBW*                              accumBuffer = nullptr;
    };

    namespace Device
    {
        class ViewportRenderer : public Device::GenericObject
        {
            friend Host::ViewportRenderer;

        public:
            __host__ __device__ ViewportRenderer() {}
            
            //__device__ void Prepare(const uint dirtyFlags);
            __device__ void Render();
            __device__ void Composite(Device::ImageRGBA* outputImage);

            __device__ void Synchronise(const ViewportRendererParams& params) { m_params = params; }
            __device__ void Synchronise(const ViewportObjects& objects) { objects.Validate(); m_objects = objects; }

        private:
            ViewportRendererParams      m_params;
            ViewportObjects             m_objects;
        };
    }

    namespace Host
    {        
        using DrawableObjectContainer = Host::AssetVector<Host::DrawableObject, Device::DrawableObject>;
        
        class ViewportRenderer : public Host::GenericObject
        {
        public:
            __host__ ViewportRenderer(const Asset::InitCtx& initCtx, AssetHandle<Host::GenericObjectContainer>& objectContainer, const uint width, const uint height, cudaStream_t renderStream);
            __host__ virtual ~ViewportRenderer() noexcept;

            __host__ void                       ReleaseObjects();
            __host__ virtual bool               Rebuild() override final;
            __host__ void                       Render();
            __host__ void                       Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const;

            __host__ virtual bool               Prepare(const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx, const int frameIdx);

            __host__ static const std::string   GetAssetClassStatic() { return "overlaylayer"; }
            __host__ virtual std::string        GetAssetClass() const override final { return GetAssetClassStatic(); }

            __host__ Host::BIH2DAsset& DrawableBIH() { DAssert(m_hostDrawableBIH); return *m_hostDrawableBIH; }
            __host__ Host::DrawableObjectContainer& DrawableObjects() { return *m_hostDrawableObjects; }

        protected:
            __host__ virtual void               Synchronise(const uint syncFlags) override final;
            __host__ virtual void               OnDirty(const DirtinessEvent& flag, WeakAssetHandle<Host::Asset>& caller) override final;
            __host__ void                       Summarise() const;

        private:
            AssetHandle<Host::GenericObjectContainer>& m_objectContainer;

            Device::ViewportRenderer*       cu_deviceInstance = nullptr;
            Device::ViewportRenderer        m_hostInstance;
            ViewportRendererParams          m_params;

            AssetHandle<Host::ImageRGBW>    m_hostAccumBuffer;
            dim3                            m_blockSize, m_gridSize;

            AssetHandle<Host::BIH2DAsset>               m_hostDrawableBIH;
            AssetHandle<DrawableObjectContainer>        m_hostDrawableObjects; 

            int                             m_viewportFrameIdx;
        };
    }
}