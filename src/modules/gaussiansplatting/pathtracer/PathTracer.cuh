#pragma once

#include "core/2d/Ctx.cuh"

#include "core/DirtinessFlags.cuh"
#include "core/Image.cuh"
#include "core/GenericObject.cuh"

namespace Enso
{         
    namespace Device { class Tracable; }
    
    struct PathTracerParams
    {
        __host__ __device__ PathTracerParams();
        __device__ void Validate() const;

        struct
        {
            ivec2 dims;
        } 
        viewport;

        int frameIdx;
    };

    struct PathTracerObjects
    {
        __device__ void Validate() const
        {
            //CudaAssert(scene);
            CudaAssert(accumBuffer);
        }
        
        //const Device::SceneContainer*     scene = nullptr;
        Device::ImageRGBW*                  accumBuffer = nullptr;
    };

    namespace Device
    {
        class PathTracer : public Device::GenericObject
        {
        public:
            __host__ __device__ PathTracer() {}
            
            __device__ void Prepare(const uint dirtyFlags);
            __device__ void Render();
            __device__ void Composite(Device::ImageRGBA* outputImage);

            __device__ void Synchronise(const PathTracerParams& params) { m_params = params; }
            __device__ void Synchronise(const PathTracerObjects& objects) { objects.Validate(); m_objects = objects; }

        private:
            PathTracerParams              m_params;
            PathTracerObjects             m_objects;

            //Device::SceneContainer        m_scene;
        };
    }

    namespace Host
    {
        class PathTracer : public Host::GenericObject
        {
        public:
            PathTracer(const Asset::InitCtx& initCtx, /*const AssetHandle<const Host::SceneContainer>& scene, */const uint width, const uint height, cudaStream_t renderStream);
            virtual ~PathTracer() noexcept;

            __host__ void Render();
            __host__ void Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const; 

            __host__ bool Prepare();
            __host__ void Clear();

            __host__ static const std::string  GetAssetClassStatic() { return "pathtracer"; }
            __host__ virtual std::string       GetAssetClass() const override final { return GetAssetClassStatic(); }

        protected:
            __host__ virtual void Synchronise(const uint syncFlags) override final;

        private:
            template<typename T>
            __host__ void           KernelParamsFromImage(const AssetHandle<Host::Image<T>>& image, dim3& blockSize, dim3& gridSize) const
            {
                const auto& meta = image->GetMetadata();
                blockSize = dim3(16, 16, 1);
                gridSize = dim3((meta.Width() + 15) / 16, (meta.Height() + 15) / 16, 1);
            }

            //const AssetHandle<const Host::SceneContainer>& m_scene;

            Device::PathTracer*               cu_deviceInstance = nullptr;
            Device::PathTracer                m_hostInstance;
            PathTracerObjects                 m_deviceObjects;
            PathTracerParams                  m_params;

            AssetHandle<Host::ImageRGBW>        m_hostAccumBuffer;
        };
    }
}