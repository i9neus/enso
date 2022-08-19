#pragma once

#include "UICtx.cuh"

#include "../CudaAsset.cuh"
#include "../CudaManagedObject.cuh"
#include "../CudaImage.cuh"
#include "../CudaVector.cuh"

using namespace Cuda;

namespace GI2D
{
    namespace Host
    {
        class UILayer : public Cuda::Host::Asset
        {
        public:
            UILayer(const std::string& id) : Asset(id) {}
            virtual ~UILayer() = default;

            __host__ virtual void   Render() = 0;
            __host__ virtual void   Composite(AssetHandle<Cuda::Host::ImageRGBA>& hostOutputImage) const = 0;
            __host__ virtual void   Synchronise() = 0;

            __host__ void           SetViewCtx(const UIViewCtx& ctx) { m_viewCtx = ctx; m_isDirty = true; }
            __host__ void           SetDirty() { m_isDirty = true; }

        protected:
            template<typename T>
            __host__ void           KernelParamsFromImage(const AssetHandle<Cuda::Host::Image<T>>& image, dim3& blockSize, dim3& gridSize) const
            {
                const auto& meta = image->GetMetadata();
                blockSize = dim3(16, 16, 1);
                gridSize = dim3((meta.Width() + 15) / 16, (meta.Height() + 15) / 16, 1);
            }

        protected:
            UIViewCtx   m_viewCtx;

            bool        m_isDirty;
        };
    }
}