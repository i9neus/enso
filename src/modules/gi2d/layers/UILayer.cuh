#pragma once

#include "../UICtx.cuh"

#include "../Common.cuh"
#include "core/Image.cuh"

#include "../tracables/Tracable.cuh"
#include "../FwdDecl.cuh"
#include "../SceneDescription.cuh"

namespace Enso
{
    struct UILayerParams
    {
        __host__ __device__ UILayerParams()
        {
            selectionCtx.isLassoing = false;
        }
        
        UIViewCtx           viewCtx;
        UISelectionCtx      selectionCtx;
    };

    namespace Device
    {
        class UILayer : public Device::Asset
        {
        public:
            __device__ UILayer() {}

            __device__ void Synchronise(const UILayerParams& params) { m_params = params; }
            __host__ __device__ virtual void OnSynchronise(const int) {};

        protected:
            __device__ __forceinline__ const UISelectionCtx& GetSelectionCtx() const { return m_params.selectionCtx; }

            UILayerParams m_params;
        };
    }

    namespace Host
    {        
        template<typename, typename> class AssetVector;
        class BIH2DAsset;
        
        class UILayer : public Host::AssetAllocator
        {
        public:
            UILayer(const std::string& id, const AssetHandle<Host::SceneDescription>& scene) :
                AssetAllocator(id),
                m_scene(scene)
            {
            }

            virtual ~UILayer() = default;

            __host__ virtual void   OnPreRender() {}
            __host__ virtual void   Render() = 0;
            __host__ virtual void   Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const = 0;

            __host__ void Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx)
            { 
                m_params.viewCtx = viewCtx;  
                m_params.selectionCtx = selectionCtx;
                m_dirtyFlags = dirtyFlags;
            }

            __host__ void           SetDirtyFlags(const uint flags) { m_dirtyFlags |= flags; }

        protected:
            template<typename SubType>
            __host__ void Synchronise(SubType* cu_object, const int syncType)
            {
                if (syncType & kSyncParams) { SynchroniseObjects<Device::UILayer>(cu_object, m_params); }
            }
            
            template<typename T>
            __host__ void           KernelParamsFromImage(const AssetHandle<Host::Image<T>>& image, dim3& blockSize, dim3& gridSize) const
            {
                const auto& meta = image->GetMetadata();
                blockSize = dim3(16, 16, 1);
                gridSize = dim3((meta.Width() + 15) / 16, (meta.Height() + 15) / 16, 1);
            }

        protected:   
            AssetHandle<Host::SceneDescription>                     m_scene;
            UILayerParams                                           m_params;

            uint                                                    m_dirtyFlags;
        };
    }
}