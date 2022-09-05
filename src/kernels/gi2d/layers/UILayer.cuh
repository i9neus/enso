#pragma once

#include "../UICtx.cuh"

#include "../Common.cuh"
#include "kernels/CudaManagedObject.cuh"
#include "kernels/CudaImage.cuh"

#include "../tracables/Tracable.cuh"
#include "kernels/gi2d/FwdDecl.cuh"

using namespace Cuda;

//namespace Cuda
//{
//    namespace Host { template<typename T> class AssetVector; }
//}

namespace Cuda { namespace Host { template<typename, typename> class AssetVector; } }
namespace GI2D { namespace Host { class BIH2DAsset; } }

namespace GI2D
{
    namespace Host
    {        
        class UILayer : public Cuda::Host::AssetAllocator
        {
        public:
            UILayer(const std::string& id, AssetHandle<GI2D::Host::BIH2DAsset>& bih, AssetHandle<TracableContainer>& tracables) :
                AssetAllocator(id),
                m_hostBIH(bih),
                m_hostTracables(tracables)
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

                RebuildImpl();
            }

            __host__ void           SetDirtyFlags(const uint flags) { m_dirtyFlags |= flags; }

        protected:
            __host__ virtual void RebuildImpl() = 0;
            
            template<typename T>
            __host__ void           KernelParamsFromImage(const AssetHandle<Cuda::Host::Image<T>>& image, dim3& blockSize, dim3& gridSize) const
            {
                const auto& meta = image->GetMetadata();
                blockSize = dim3(16, 16, 1);
                gridSize = dim3((meta.Width() + 15) / 16, (meta.Height() + 15) / 16, 1);
            }

        protected:
            UIViewCtx                                               m_viewCtx;
            UISelectionCtx                                          m_selectionCtx;
             
            AssetHandle<GI2D::Host::BIH2DAsset>                     m_hostBIH;
            AssetHandle<TracableContainer>    m_hostTracables;

            uint                                                    m_dirtyFlags;
        };
    }
}