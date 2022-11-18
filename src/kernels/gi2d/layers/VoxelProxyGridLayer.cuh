#pragma once

#include "UILayer.cuh"
#include "../integrators/PathTracer2D.cuh"
#include "../integrators/Camera2D.cuh"
#include "../SceneDescription.cuh"

using namespace Cuda;

namespace GI2D
{     
    struct VoxelProxyGridLayerParams
    {
        __host__ __device__ VoxelProxyGridLayerParams();

        BidirectionalTransform2D        m_cameraTransform;

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
        m_grid;
    };

    struct VoxelProxyGridLayerObjects
    {
        const Device::SceneDescription* m_scenePtr = nullptr;
        
        Core::Device::Vector<vec3>* m_accumBuffer = nullptr;
        Core::Device::Vector<vec3>* m_reduceBuffer = nullptr;
        Core::Device::Vector<vec3>* m_gridBuffer = nullptr;
    };

    namespace Device
    {
        class VoxelProxyGridLayer : public UILayer,
                                public VoxelProxyGridLayerParams,
                                public VoxelProxyGridLayerObjects,
                                public Camera2D
        {
        public:
            __device__ VoxelProxyGridLayer() : m_voxelTracer(m_scene) {}

            __device__ __forceinline__ void Prepare(const uint dirtyFlags);
            __device__ __forceinline__ void Render();
            __device__ __forceinline__ void Composite(Cuda::Device::ImageRGBA* outputImage) const;
            __device__ __forceinline__ vec3 Evaluate(const vec2& posWorld) const;

            __device__ virtual bool CreateRay(Ray2D& ray, HitCtx2D& hit, RenderCtx& renderCtx) const override final;
            __device__ virtual void Accumulate(const vec4& L, const RenderCtx& ctx) override final;

            __device__ virtual void OnSynchronise(const int) override final;

            __device__ void ReduceAccumulationBuffer(const uint batchSize, const uvec2 batchRange);

        private:
            PathTracer2D                            m_voxelTracer;
            int                                     m_frameIdx;

            Device::SceneDescription                m_scene;
        };
    }

    namespace Host
    {
        class VoxelProxyGridLayer : public UILayer,
                                    public VoxelProxyGridLayerParams
        {
        public:
            VoxelProxyGridLayer(const std::string& id, AssetHandle<Host::SceneDescription>& scene, const uint width, const uint height);
            virtual ~VoxelProxyGridLayer();
           
            __host__ virtual void Render() override final;
            __host__ virtual void Composite(AssetHandle<Cuda::Host::ImageRGBA>& hostOutputImage) const override final;
            __host__ void OnDestroyAsset();

            __host__ void Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx);

        protected:
            __host__ void Synchronise(const int syncType);

            __host__ void Integrate();

        private:
            GI2D::Device::VoxelProxyGridLayer*          cu_deviceInstance = nullptr;
            VoxelProxyGridLayerObjects                  m_deviceObjects;

            AssetHandle<Core::Host::Vector<vec3>>       m_hostAccumBuffer;
            AssetHandle<Core::Host::Vector<vec3>>       m_hostReduceBuffer;
            AssetHandle<Core::Host::Vector<vec3>>       m_hostProxyGrid;

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