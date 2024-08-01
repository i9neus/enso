#pragma once

#include "core/2d/Ctx.cuh"

#include "core/DirtinessFlags.cuh"
#include "core/Image.cuh"
#include "core/GenericObject.cuh"
#include "core/HighResolutionTimer.h"

#include "../FwdDecl.cuh"

namespace Enso
{
    struct NLMDenoiserParams
    {
        __host__ __device__ NLMDenoiserParams();
        __device__ void Validate() const;

        struct
        {
            ivec2 dims;
        }
        viewport;

        int frameIdx;
        float wallTime;
    };

    struct NLMDenoiserObjects
    {
        __device__ void Validate() const;

        //const Device::SceneContainer*     scene = nullptr;
        Device::ImageRGBW* accumBuffer = nullptr;
        Device::Vector<BidirectionalTransform>* transforms = nullptr;
    };
    
    namespace Device
    {
        class NLMDenoiser : public Device::GenericObject
        {
        public:
            __host__ __device__ NLMDenoiser() {}

            //__device__ void Prepare(const uint dirtyFlags);
            __device__ void Render();
            __device__ void Composite(Device::ImageRGBA* outputImage);

            __device__ void Synchronise(const NLMDenoiserParams& params) { m_params = params; }
            __device__ void Synchronise(const NLMDenoiserObjects& objects) { objects.Validate(); m_objects = objects; }

        private:
            NLMDenoiserParams              m_params;
            NLMDenoiserObjects             m_objects;

            //Device::SceneContainer        m_scene;
        };
    }

    namespace Host
    {
        class NLMDenoiser : public Host::GenericObject
        {
        public:
            NLMDenoiser(const Asset::InitCtx& initCtx, /*const AssetHandle<const Host::SceneContainer>& scene, */const uint width, const uint height, cudaStream_t renderStream);
            virtual ~NLMDenoiser() noexcept;

            __host__ void Render();
            __host__ void Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const;

            __host__ bool Prepare();
            __host__ void Clear();

            __host__ static const std::string  GetAssetClassStatic() { return "pathtracer"; }
            __host__ virtual std::string       GetAssetClass() const override final { return GetAssetClassStatic(); }

        protected:
            __host__ virtual void Synchronise(const uint syncFlags) override final;

        private:
            __host__ void CreateScene();

            template<typename T>
            __host__ void           KernelParamsFromImage(const AssetHandle<Host::Image<T>>& image, dim3& blockSize, dim3& gridSize) const
            {
                const auto& meta = image->GetMetadata();
                blockSize = dim3(16, 16, 1);
                gridSize = dim3((meta.Width() + 15) / 16, (meta.Height() + 15) / 16, 1);
            }

            //const AssetHandle<const Host::SceneContainer>& m_scene;

            Device::NLMDenoiser* cu_deviceInstance = nullptr;
            Device::NLMDenoiser                m_hostInstance;
            NLMDenoiserObjects                 m_deviceObjects;
            NLMDenoiserParams                  m_params;
            HighResolutionTimer               m_wallTime;

            AssetHandle<Host::ImageRGBW>        m_hostAccumBuffer;
            AssetHandle<Host::Vector<BidirectionalTransform>> m_hostTransforms;
        };
    }
}