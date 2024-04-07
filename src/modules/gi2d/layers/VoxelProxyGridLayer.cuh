#pragma once

#include "../FwdDecl.cuh""
#include "UILayer.cuh"
#include "../integrators/Camera2D.cuh"
#include "../integrators/AccumulationBuffer.cuh"

namespace Enso
{     
    struct VoxelProxyGridLayerParams
    {
        __host__ __device__ VoxelProxyGridLayerParams() { gridSize = ivec2(0);  }
        
        __device__ void Validate() const 
        {
            CudaAssert(gridSize.x > 0 && gridSize.y > 0);            

            accum.Validate();
        }

        BidirectionalTransform2D        cameraTransform;
        UIViewCtx                       viewCtx;

        ivec2                           gridSize;
        AccumulationBufferParams        accum;
    };

    struct VoxelProxyGridLayerObjects
    {
        __device__ void Validate() const
        {
            CudaAssert(scene);
            CudaAssert(accumBuffer);
        }
        
        Device::AccumulationBuffer*         accumBuffer = nullptr;
        const Device::SceneDescription*     scene = nullptr;  
    };

    namespace Device
    {       
        class VoxelProxyGridLayer : public Device::GenericObject, public Device::Camera2D
        {
        public:
            __device__ VoxelProxyGridLayer() {}

            __device__ __forceinline__ void     Render();
            __device__ __forceinline__ void     Composite(Device::ImageRGBA* outputImage) const;
            __device__ __forceinline__ vec3     Evaluate(const vec2& posWorld) const;

            __device__ virtual bool             CreateRay(Ray2D& ray, HitCtx2D& hit, RenderCtx& renderCtx) const override final;
            __device__ virtual void             Accumulate(const vec4& L, const RenderCtx& ctx) override final;

            //__host__ __device__ virtual void    OnSynchronise(const int) override final;
            __device__ void                     Synchronise(const VoxelProxyGridLayerParams& params) { m_params = params; }
            __device__ void                     Synchronise(const VoxelProxyGridLayerObjects& objects);

        private:
            PathTracer2D                            m_voxelTracer;
            int                                     m_frameIdx;

            VoxelProxyGridLayerParams               m_params;
            VoxelProxyGridLayerObjects              m_objects;
            Device::SceneDescription                m_scene;
        };
    }

    namespace Host
    {
        class VoxelProxyGridLayer : public Host::UILayer, public Host::GenericObject, public Host::Camera2D
        {
        public:
            VoxelProxyGridLayer(const std::string& id, const Json::Node& json, const AssetHandle<const Host::SceneDescription>& scene);
            virtual ~VoxelProxyGridLayer();
           
            __host__ virtual void Render() override final;
            __host__ virtual bool Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx) override final;
            __host__ virtual void Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const override final;
            __host__ void OnDestroyAsset();

            //__host__ static AssetHandle<Host::GenericObject> Instantiate(const std::string& id, const Json::Node&, const AssetHandle<const Host::SceneDescription>& scene);
            __host__ static const std::string  GetAssetClassStatic() { return "voxelproxygridlayer"; }
            __host__ virtual std::string       GetAssetClass() const override final { return GetAssetClassStatic(); }

            __host__ virtual Device::VoxelProxyGridLayer* GetDeviceInstance() const override final { return cu_deviceInstance; }

            __host__ virtual bool Serialise(Json::Node& rootNode, const int flags) const override;
            __host__ virtual uint Deserialise(const Json::Node& rootNode, const int flags) override;

        protected:
            __host__ void Synchronise(const int syncType);

        private:
            Device::VoxelProxyGridLayer*            cu_deviceInstance = nullptr;
            Device::VoxelProxyGridLayer             m_hostInstance;
            VoxelProxyGridLayerObjects              m_deviceObjects;
            VoxelProxyGridLayerParams               m_params;

            AssetHandle<Host::AccumulationBuffer>   m_accumBuffer;

            const AssetHandle<const Host::SceneDescription>& m_scene;
        };

    }
}