#pragma once

#include "../UICtx.cuh"

#include "../Common.cuh"
#include "kernels/CudaManagedObject.cuh"
#include "kernels/CudaImage.cuh"

#include "../tracables/Tracable.cuh"
#include "kernels/gi2d/FwdDecl.cuh"
#include "../SceneDescription.cuh"

using namespace Cuda;

//namespace Cuda
//{
//    namespace Host { template<typename T> class AssetVector; }
//}

namespace Cuda { namespace Host { template<typename, typename> class AssetVector; } }
namespace GI2D { namespace Host { class BIH2DAsset; } }

namespace GI2D
{
    struct UILayerParams
    {
        __host__ __device__ UILayerParams()
        {
            m_selectionCtx.isLassoing = false;
        }
        
        UIViewCtx           m_viewCtx;
        UISelectionCtx      m_selectionCtx;
    };

    namespace Device
    {
        class UILayer : public Cuda::Device::Asset,
                        public UILayerParams
        {
        public:
            __device__ UILayer() {}

            __device__ virtual void OnSynchronise(const int) {};
        };
    }

    namespace Host
    {        
        class UILayer : public Cuda::Host::AssetAllocator,
                        public UILayerParams
        {
        public:
            UILayer(const std::string& id, const AssetHandle<Host::SceneDescription>& scene) :
                AssetAllocator(id),
                m_scene(scene)
            {
            }

            virtual ~UILayer() = default;

            __host__ virtual void   Render() = 0;
            __host__ virtual void   Composite(AssetHandle<Cuda::Host::ImageRGBA>& hostOutputImage) const = 0;

            __host__ void Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx)
            { 
                m_viewCtx = viewCtx;  
                m_selectionCtx = selectionCtx;
                m_dirtyFlags = dirtyFlags;
            }

            __host__ void           SetDirtyFlags(const uint flags) { m_dirtyFlags |= flags; }

        protected:
            template<typename SubType>
            __host__ void Synchronise(SubType* cu_object, const int syncType)
            {
                if (syncType & kSyncParams)  { SynchroniseInheritedClass<UILayerParams>(cu_object, *this, kSyncParams); }
            }
            
            template<typename T>
            __host__ void           KernelParamsFromImage(const AssetHandle<Cuda::Host::Image<T>>& image, dim3& blockSize, dim3& gridSize) const
            {
                const auto& meta = image->GetMetadata();
                blockSize = dim3(16, 16, 1);
                gridSize = dim3((meta.Width() + 15) / 16, (meta.Height() + 15) / 16, 1);
            }

        protected:   
            AssetHandle<Host::SceneDescription>                     m_scene;

            uint                                                    m_dirtyFlags;
        };
    }
}