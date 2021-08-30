#pragma once

#include "CudaTracable.cuh"

#define kSDFMaxIterations 10

namespace Json { class Node; }

namespace Cuda
{
    template<int NumVertices, int NumFaces, int NumEdges, int PolyOrder>
    struct SimplePolyhedron
    {
        enum _attrs : int { kNumVertices = NumVertices, 
                            kNumFaces = NumFaces, 
                            kNumEdges = NumEdges,
                            kPolyOrder = PolyOrder };

        __device__ SimplePolyhedron() {}

        __device__ void Prepare(const float scale)
        {
            // Scale the vertices
            for (int vertIdx = 0; vertIdx < NumVertices; ++vertIdx) { V[vertIdx] *= scale; }
            
            // Pre-compute deltas and face and edge normals
            for (int faceIdx = 0, offset = 0; faceIdx < NumFaces; ++faceIdx, offset += PolyOrder)
            {
                N[faceIdx] = normalize(cross(V[F[offset + 1]] - V[F[offset + 0]], V[F[offset + 2]] - V[F[offset + 0]]));

                for (int edge = 0; edge < PolyOrder; ++edge)
                {
                    dV[offset + edge] = (V[F[offset + (edge + 1) % PolyOrder]] - V[F[offset + edge]]);
                    edgeNorm[offset + edge] = normalize(cross(dV[offset + edge], N[faceIdx]));
                }
            }
        }

        vec3		V[NumVertices];
        uchar		F[NumFaces * PolyOrder];
        uchar       E[NumEdges * 2];

        vec3        N[NumFaces];
        vec3        dV[NumFaces * PolyOrder];
        vec3        edgeNorm[NumFaces * PolyOrder];

        float		sqrBoundRadius;
    };
    
    // Foreward declarations
    namespace Host { class KIFS; }    
    namespace SDF { struct PolyhedronData; }
    struct BlockConstantData;

    enum KIFSPrimitive : int 
    { 
        kKIFSPrimitiveTetrahedronSolid,
        KIFSPrimitiveCubeSolid,
        KIFSPrimitiveSphere, 
        KIFSPrimitiveTorus, 
        KIFSPrimitiveBox,
        kKIFSPrimitiveTetrahedronCage,
        KIFSPrimitiveCubeCage
    };
    enum KIFSClipShape : int 
    { 
        kKIFSClipBox, kKIFSClipSphere, kKIFSClipTorus 
    };
    enum KIFSFold : int
    {
        kKIFSFoldTetrahedron,
        kKIFSFoldCube
    };

    struct KIFSParams
    {
        __host__ __device__ KIFSParams();
        __host__ KIFSParams(const ::Json::Node& node, const uint flags);

        __host__ void ToJson(::Json::Node& node) const;
        __host__ void FromJson(const ::Json::Node& node, const uint flags);
        __host__ void Randomise(const vec2& range);

        JitterableFloat     rotateA;
        JitterableFloat     rotateB;
        JitterableFloat     scaleA;
        JitterableFloat     scaleB;
        JitterableFloat     vertScale;
        JitterableFloat     crustThickness;
        JitterableFlags     faceMask;

        int                 numIterations;
        int                 foldType;
        int                 primitiveType;

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

                SimplePolyhedron<4, 4, 6, 3>   tetrahedronData;
                SimplePolyhedron<8, 6, 12, 4>   cubeData;

                vec3 xi[kSDFMaxIterations];
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
            KIFSParams m_params;

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
            __host__ virtual const RenderObjectParams* GetRenderObjectParams() const override final { return &m_params.tracable.renderObject; }
        };
    }
}