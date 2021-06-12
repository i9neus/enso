#pragma once

#include "CudaTracable.cuh"

namespace Cuda
{
    namespace Host { class KIFS; }

    enum class KIFSType : uint { kTetrahedtron, kCube };

    namespace Device
    {
        class KIFS : public Device::Tracable
        {
            friend Host::KIFS;
        public: 
            struct Params
            {
                //__host__ __device__ Params() : albedo(0.5f) {}

                //vec3 ;
            }
            m_params;

        protected:
            KIFS() = default;

            __device__ vec4 Field(vec3 p, const mat3& b, const RenderCtx& renderCtx, uint& code, uint& surfaceDepth) const;
            __device__ void FoldTetrahedron(const mat4& matrix, const int& i, vec3& p, mat3& bi, uint& code) const; 

            KIFSType    m_type;
            vec3*       m_V;
            uint*       m_F;
            vec3        m_origin;

            uint        m_numVertices;
            uint        m_numFaces;
            uint        m_polyOrder;

        public:
            __device__ KIFS(const BidirectionalTransform& transform);
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
            Device::KIFS  m_hostData;

        public:
            __host__ KIFS();
            __host__ virtual ~KIFS() { OnDestroyAsset(); }
            __host__ virtual void OnDestroyAsset() override final;
            __host__ virtual void OnJson(const Json::Node& jsonNode) override final;

            __host__ virtual Device::KIFS* GetDeviceInstance() const override final { return cu_deviceData; }
        };
    }
}