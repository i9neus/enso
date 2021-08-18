#pragma once

#include "CudaTracable.cuh"

#define kSDFMaxIterations 10

namespace Json { class Node; }

namespace Cuda
{
    // Foreward declarations
    namespace Host { class KIFS; }    
    namespace SDF { struct PolyhedronData; }
    struct BlockConstantData;

    enum KIFSPrimitive : int { kKIFSTetrahedtron, kKIFSCube };
    enum KIFSClipShape : int { kKIFSBox, kKIFSSphere, kKIFSTorus };

    struct KIFSParams
    {
        __host__ __device__ KIFSParams();
        __host__ KIFSParams(const ::Json::Node& node, const uint flags);

        __host__ void ToJson(::Json::Node& node) const;
        __host__ void FromJson(const ::Json::Node& node, const uint flags);

        vec3    rotateA;
        vec3    rotateB;
        vec3    scaleA;
        vec3    scaleB;
        vec3    vertScale;
        vec3    crustThickness;

        int     numIterations;
        uint    faceMask;
        int     foldType;
        int     primitiveType;

        struct
        {
            int maxSpecularIterations;
            int maxDiffuseIterations;
            float cutoffThreshold;
            float escapeThreshold;
            float rayIncrement;
            float failThreshold;
            float rayKickoff;
            
            bool clipCameraRays;
            int clipShape;
        }
        sdf;

        TracableParams tracable;
        BidirectionalTransform transform;

        bool doTakeSnapshot;
    };

    namespace Device
    {
        class KIFS : public Device::Tracable
        {
            friend Host::KIFS;
        public:
            struct KernelConstantData
            {
                int                         numIterations;
                float                       crustThickness;
                float                       vertScale;
                uint                        faceMask;
                int                         foldType;
                int                         primitiveType;

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
            __device__ void Prepare();

            vec3        m_origin;

            KIFSParams      m_params;
            KernelConstantData m_kernelConstantData;

            uint        m_numVertices;
            uint        m_numFaces;
            uint        m_polyOrder;

        public:
            __device__ KIFS();
            __device__ ~KIFS() {}

            __device__ virtual bool Intersect(Ray& ray, HitCtx& hit) const override final;
            __device__ virtual void InitialiseKernelConstantData() const override final;
            __device__ void Synchronise(const KIFSParams& params)
            { 
                m_params = params;
                Prepare();
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
            __host__ KIFS(const ::Json::Node& node);
            __host__ virtual ~KIFS() = default;

            __host__ static AssetHandle<Host::RenderObject> Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);

            __host__ virtual void OnDestroyAsset() override final;
            __host__ virtual void FromJson(const ::Json::Node& node, const uint flags) override final;
            __host__ static std::string GetAssetTypeString() { return "kifs"; }
            __host__ static std::string GetAssetDescriptionString() { return "KIFS Fractal"; }
            __host__ virtual Device::KIFS* GetDeviceInstance() const override final { return cu_deviceData; }
            __host__ virtual int GetIntersectionCostHeuristic() const override final { return 100; };
        };
    }
}