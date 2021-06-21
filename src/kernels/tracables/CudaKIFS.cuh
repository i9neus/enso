#pragma once

#include "CudaTracable.cuh"

#define kSDFMaxIterations 10

namespace Cuda
{
    // Foreward declarations
    namespace Host { class KIFS; }    
    namespace SDF { struct PolyhedronData; }
    struct BlockConstantData;

    enum class KIFSType : uint { kTetrahedtron, kCube };

    namespace Device
    {        
        class KIFS : public Device::Tracable
        {
            friend Host::KIFS;
        public: 
            struct Params
            {
                vec3    rotate;
                vec2    scale;

                float   vertScale;
                float   thickness;
                int     iterations;
                int     maxIterations;
                uint    faceMask;
            }; 

            struct KernelConstantData
            {
                int                         numIterations;
                float                       isosurfaceThickness;
                float                       vertScale;
                uint                        faceMask;

                struct
                {
                    mat3                    matrix;
                    float                   scale;
                }
                iteration[kSDFMaxIterations];
            };

            __device__ static uint SizeOfSharedMemory();

        protected:
            __device__ vec4 Field(vec3 p, const mat3& b, uint& code, uint& surfaceDepth) const;
            __device__ void Initialise();
            __device__ __inline__ void PrepareKernelConstantData(KernelConstantData& bd) const;

            KIFSType    m_type;
            vec3        m_origin;

            Params      m_params;
            KernelConstantData m_kernelConstantData;

            mat3        m_matrices[kSDFMaxIterations];
            float       m_iterScales[kSDFMaxIterations];
            ivec2       m_iterations;
            int         m_maxIterations;
            float       m_isosurfaceThickness;
            float       m_vertScale;
            uint        m_faceMask;

            uint        m_numVertices;
            uint        m_numFaces;
            uint        m_polyOrder;           

        public:
            __device__ KIFS();
            __device__ ~KIFS() = default;

            __device__ bool Intersect(Ray& ray, HitCtx& hit) const;
            __device__ void Precache();
            __device__ void OnSyncParameters(const Params& params) 
            { 
                m_params = params;
                Initialise();
            }
        };
    }

    namespace Host
    {
        class KIFS : public Host::Tracable
        {
        private:
            Device::KIFS* cu_deviceData;

        public:
            __host__ KIFS();
            __host__ virtual ~KIFS() { OnDestroyAsset(); }
            __host__ virtual void OnDestroyAsset() override final;
            __host__ virtual void OnJson(const Json::Node& jsonNode) override final;

            __host__ virtual Device::KIFS* GetDeviceInstance() const override final { return cu_deviceData; }
        };
    }
}