#pragma once

#include "UILayer.cuh"
#include "../integrators/PathTracer2D.cuh"
#include "../integrators/Camera2D.cuh"
#include "../SceneDescription.cuh"
#include "../tracables/KIFS.cuh"

namespace Enso
{     
    struct VoxelProxyGridLayerParams
    {
        __host__ __device__ VoxelProxyGridLayerParams();
        __device__ void Validate() const {}

        BidirectionalTransform2D        cameraTransform;
        UIViewCtx                       viewCtx;

        struct
        {
            ivec2           size;
            int             numProbes;
            int             numHarmonics;

            int             totalGridUnits; //              <-- The total number of units in the reduced grid
            int	    		subprobesPerProbe;	//			<-- A sub-probe is a set of SH coefficients + data. Multiple sub-probes are accumulated to make a full probe. 
            int             unitsPerProbe; //				<-- The total number of accumulation units (coefficients + data) accross all sub-probes, per probe
            int             totalAccumUnits; //		        <-- The total number of accumulation units in the grid
            int             totalSubprobes; //				<-- The total number of subprobes in the grid
        }
        grid;
    };

    struct VoxelProxyGridLayerObjects
    {
        __device__ void Validate() const
        {
            assert(scenePtr);
            assert(accumBuffer);
            assert(reduceBuffer);
            assert(gridBuffer);
        }
        
        const Device::SceneDescription*     scenePtr = nullptr;        
        Device::Vector<vec3>*               accumBuffer = nullptr;
        Device::Vector<vec3>*               reduceBuffer = nullptr;
        Device::Vector<vec3>*               gridBuffer = nullptr;
    };

    namespace Device
    {
        class VoxelProxyGridLayer : public Device::ICamera2D, 
                                    public Device::GenericObject
        {
        public:
            __device__ VoxelProxyGridLayer() : m_voxelTracer(m_scene) {}

            __device__ __forceinline__ void     Prepare(const uint dirtyFlags);
            __device__ __forceinline__ void     Render();
            __device__ __forceinline__ void     Composite(Device::ImageRGBA* outputImage) const;
            __device__ __forceinline__ vec3     Evaluate(const vec2& posWorld) const;

            __device__ virtual bool             CreateRay(Ray2D& ray, HitCtx2D& hit, RenderCtx& renderCtx) const override final;
            __device__ virtual void             Accumulate(const vec4& L, const RenderCtx& ctx) override final;

            __host__ __device__ virtual void    OnSynchronise(const int) override final;
            __device__ void                     Synchronise(const VoxelProxyGridLayerParams& params) { m_params = params; }
            __device__ void                     Synchronise(const VoxelProxyGridLayerObjects& objects) { objects.Validate(); m_objects = objects; }

            __device__ void ReduceAccumulationBuffer(const uint batchSize, const uvec2 batchRange);

        private:
            PathTracer2D                            m_voxelTracer;
            int                                     m_frameIdx;

            VoxelProxyGridLayerParams               m_params;
            VoxelProxyGridLayerObjects              m_objects;
            Device::SceneDescription                m_scene;

            KIFSDebugData                           m_kifsDebug;
        };
    }

    namespace Host
    {
        class VoxelProxyGridLayer : public Host::UILayer, 
                                    public Host::GenericObject
        {
        public:
            VoxelProxyGridLayer(const std::string& id, const Json::Node& json, const AssetHandle<const Host::SceneDescription>& scene);
            virtual ~VoxelProxyGridLayer();
           
            __host__ virtual void Render() override final;
            __host__ virtual void Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx) override final;
            __host__ virtual void Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const override final;
            __host__ void OnDestroyAsset();

            //__host__ static AssetHandle<Host::GenericObject> Instantiate(const std::string& id, const Json::Node&, const AssetHandle<const Host::SceneDescription>& scene);
            __host__ static const std::string  GetAssetClassStatic() { return "voxelproxygridlayer"; }
            __host__ virtual std::string       GetAssetClass() const override final { return GetAssetClassStatic(); }

            //__host__ virtual Device::VoxelProxyGridLayer* GetDeviceInstance() const override final { return cu_deviceInstance; }

            __host__ virtual bool Serialise(Json::Node& rootNode, const int flags) const override;
            __host__ virtual uint Deserialise(const Json::Node& rootNode, const int flags) override;

        protected:
            __host__ void Synchronise(const int syncType);

            __host__ void Reduce();

        private:
            Device::VoxelProxyGridLayer*            cu_deviceInstance = nullptr;
            Device::VoxelProxyGridLayer             m_hostInstance;
            VoxelProxyGridLayerObjects              m_deviceObjects;
            VoxelProxyGridLayerParams               m_params;

            AssetHandle<Host::Vector<vec3>>         m_hostAccumBuffer;
            AssetHandle<Host::Vector<vec3>>         m_hostReduceBuffer;
            AssetHandle<Host::Vector<vec3>>         m_hostProxyGrid;

            const AssetHandle<const Host::SceneDescription>& m_scene;

            struct
            {
                int blockSize;
                struct
                {
                    int accumSize;
                    int reduceSize;
                    int compSize;
                }
                grids;
            }
            m_kernelParams;
        };

    }
}