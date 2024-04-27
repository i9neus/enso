#pragma once

#include "../FwdDecl.cuh"
#include "Camera.cuh"

namespace Enso
{     
    struct VoxelProxyGridParams
    {
        __host__ __device__ VoxelProxyGridParams() { gridSize = ivec2(0);  }
        
        __device__ void Validate() const 
        {
            CudaAssert(gridSize.x > 0 && gridSize.y > 0);
        }

        BidirectionalTransform2D        cameraTransform;
        UIViewCtx                       viewCtx;

        ivec2                           gridSize;
    };

    struct VoxelProxyGridObjects
    {
        __device__ void Validate() const
        {
            CudaAssert(accumBuffer);
        }
        
        Device::AccumulationBuffer*         accumBuffer = nullptr;
    };

    namespace Device
    {       
        class VoxelProxyGrid : public Device::Camera
        {
        public:
            __host__ __device__ VoxelProxyGrid() {}

            __device__ __forceinline__ vec3     Evaluate(const vec2& posWorld) const;

            __device__ virtual bool             CreateRay(Ray2D& ray, HitCtx2D& hit, RenderCtx& renderCtx) const override final;
            __device__ virtual void             Accumulate(const vec4& L, const RenderCtx& ctx) override final;
            __host__ __device__ virtual vec4    EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx, const bool isMouseTest) const override final;

            //__host__ __device__ virtual void    OnSynchronise(const int) override final;
            __device__ void                     Synchronise(const VoxelProxyGridParams& params) { m_params = params; }
            __device__ void                     Synchronise(const VoxelProxyGridObjects& objects) { m_objects = objects; }

        private:
            VoxelProxyGridParams               m_params;
            VoxelProxyGridObjects              m_objects;
        };
    }

    namespace Host
    {
        class VoxelProxyGrid : public Host::Camera
        {
        public:
            VoxelProxyGrid(const Asset::InitCtx& initCtx, const Json::Node& json, const AssetHandle<const Host::SceneContainer>& scene);
            virtual ~VoxelProxyGrid() noexcept;
            
            __host__ virtual bool IsTransformable() const override final { return false; }

            //__host__ static AssetHandle<Host::GenericObject> Instantiate(const std::string& id, const Json::Node&, const AssetHandle<const Host::SceneContainer>& scene);
            __host__ static const std::string  GetAssetClassStatic() { return "voxelproxygrid"; }
            __host__ virtual std::string       GetAssetClass() const override final { return GetAssetClassStatic(); }

            __host__ virtual Device::VoxelProxyGrid* GetDeviceInstance() const override final { return cu_deviceInstance; }

            __host__ virtual bool Serialise(Json::Node& rootNode, const int flags) const override;
            __host__ virtual bool Deserialise(const Json::Node& rootNode, const int flags) override;

            __host__ virtual BBox2f GetObjectSpaceBoundingBox() override final { return BBox2f::MakeInvalid(); }

        protected:
            __host__ bool OnCreateSceneObject(const std::string& stateID, const UIViewCtx& viewCtx, const vec2& mousePosObject) override final { return false; }
            __host__ void Synchronise(const int syncType);

        private:
            Device::VoxelProxyGrid*            cu_deviceInstance = nullptr;
            Device::VoxelProxyGrid             m_hostInstance;
            VoxelProxyGridObjects              m_deviceObjects;
            VoxelProxyGridParams               m_params;
        };

    }
}