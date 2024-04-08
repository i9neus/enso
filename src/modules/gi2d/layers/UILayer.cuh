#pragma once

#include "../UICtx.cuh"

#include "../Dirtiness.cuh"
#include "core/Image.cuh"

#include "../tracables/Tracable.cuh"
#include "../FwdDecl.cuh"
#include "../SceneDescription.cuh"

namespace Enso
{
    namespace Host
    {        
        class UILayer
        {
        public:           
            virtual ~UILayer() = default;

            __host__ virtual void   OnPreRender() {}
            __host__ virtual void   Render() = 0;
            __host__ virtual void   Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const = 0;
            __host__ virtual bool   Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx) = 0;

        protected:
            UILayer() = default;
            
            template<typename T>
            __host__ void           KernelParamsFromImage(const AssetHandle<Host::Image<T>>& image, dim3& blockSize, dim3& gridSize) const
            {
                const auto& meta = image->GetMetadata();
                blockSize = dim3(16, 16, 1);
                gridSize = dim3((meta.Width() + 15) / 16, (meta.Height() + 15) / 16, 1);
            }
        };
    }
}