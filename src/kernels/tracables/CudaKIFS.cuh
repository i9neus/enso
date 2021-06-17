#pragma once

#include "CudaTracable.cuh"

#define kSDFMaxIterations 10

namespace Cuda
{
    namespace Host { class KIFS; }    
    namespace SDF { struct PolyhedronData; }

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
                ivec2   iterations;
                uint    faceMask;
            }
            m_params;

        protected:
            __device__ vec4 Field(vec3 p, const mat3& b, uint& code, uint& surfaceDepth) const;
            __device__ void FoldTetrahedron(const mat3& matrix, const int& i, vec3& p, mat3& bi, uint& code) const; 
            __device__ void Initialise();

            KIFSType    m_type;
            vec3        m_origin;

            const vec3  m_kXi[kSDFMaxIterations];
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

            StaticPolyhedron<4, 4, 3>     m_tetrahedronData;
            StaticPolyhedron<8, 6, 4>     m_cubeData;

        public:
            __device__ KIFS();
            __device__ ~KIFS() = default;

            __device__ bool Intersect(Ray& ray, HitCtx& hit) const;
            __device__ void OnSyncParameters(const Params& params) { m_params = params; }
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