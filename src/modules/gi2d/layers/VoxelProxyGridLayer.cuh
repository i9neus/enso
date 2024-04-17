#pragma once

#include "../FwdDecl.cuh""
#include "UILayer.cuh"
#include "../integrators/Camera.cuh"

namespace Enso
{     
    struct VoxelProxyGridLayerParams
    {
        __host__ __device__ VoxelProxyGridLayerParams() { gridSize = ivec2(0);  }
        
        __device__ void Validate() const 
        {
            CudaAssert(gridSize.x > 0 && gridSize.y > 0);
        }

        BidirectionalTransform2D        cameraTransform;
        UIViewCtx                       viewCtx;

        ivec2                           gridSize;
    };

    struct VoxelProxyGridLayerObjects
    {
        __device__ void Validate() const
        {
            CudaAssert(accumBuffer);
        }
        
        Device::AccumulationBuffer*         accumBuffer = nullptr;
    };

    namespace Device
    {       
        class VoxelProxyGridLayer : public Device::Camera
        {
        public:
            __device__ VoxelProxyGridLayer() {}

            __device__ __forceinline__ void     Composite(Device::ImageRGBA* outputImage) const;
            __device__ __forceinline__ vec3     Evaluate(const vec2& posWorld) const;

            __device__ virtual bool             CreateRay(Ray2D& ray, HitCtx2D& hit, RenderCtx& renderCtx) const override final;
            __device__ virtual void             Accumulate(const vec4& L, const RenderCtx& ctx) override final;

            //__host__ __device__ virtual void    OnSynchronise(const int) override final;
            __device__ void                     Synchronise(const VoxelProxyGridLayerParams& params) { m_params = params; }
            __device__ void                     Synchronise(const VoxelProxyGridLayerObjects& objects) { m_objects = objects; }

        private:
            VoxelProxyGridLayerParams               m_params;
            VoxelProxyGridLayerObjects              m_objects;
        };
    }

    namespace Host
    {
        class VoxelProxyGridLayer : public Host::Camera,
                                    public Host::UILayer
        {
        public:
            VoxelProxyGridLayer(const std::string& id, const Json::Node& json, const AssetHandle<const Host::SceneDescription>& scene);
            virtual ~VoxelProxyGridLayer();
            
            __host__ virtual void Render() override final;
            __host__ virtual bool Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx) override final;
            __host__ virtual void Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const override final;
            __host__ virtual bool IsTransformable() const override final { return false; }

            //__host__ static AssetHandle<Host::GenericObject> Instantiate(const std::string& id, const Json::Node&, const AssetHandle<const Host::SceneDescription>& scene);
            __host__ static const std::string  GetAssetClassStatic() { return "voxelproxygridlayer"; }
            __host__ virtual std::string       GetAssetClass() const override final { return GetAssetClassStatic(); }

            __host__ virtual Device::VoxelProxyGridLayer* GetDeviceInstance() const override final { return cu_deviceInstance; }

            __host__ virtual bool Serialise(Json::Node& rootNode, const int flags) const override;
            __host__ virtual uint Deserialise(const Json::Node& rootNode, const int flags) override;

        protected:
            __host__ void Synchronise(const int syncType);
            __host__ virtual BBox2f RecomputeObjectSpaceBoundingBox() override final { return BBox2f::MakeInvalid(); }

        private:
            Device::VoxelProxyGridLayer*            cu_deviceInstance = nullptr;
            Device::VoxelProxyGridLayer             m_hostInstance;
            VoxelProxyGridLayerObjects              m_deviceObjects;
            VoxelProxyGridLayerParams               m_params;
        };

    }
}