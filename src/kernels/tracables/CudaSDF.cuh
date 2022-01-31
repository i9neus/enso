#pragma once

#include "CudaTracable.cuh"

namespace Cuda
{    
    namespace SDFPrimitive
    {
        __device__ __forceinline__ vec4 Capsule(const vec3& p, const vec3& v0, const vec3& v1, const float r)
        {
            const vec3 dv = v1 - v0;

            float t = clamp((dot(p, dv) - dot(v0, dv)) / dot(dv, dv), 0.0f, 1.0f);

            vec3 grad = p - (v0 + t * dv);
            float gradMag = length(grad);
            grad /= gradMag;

            return vec4(gradMag - r, grad);
        }

        __device__ __forceinline__ vec4 Torus(const vec3& p, const float& r1, const float& r2)
        {
            const vec3 pPlane = vec3(p.x, 0.0f, p.z);
            float pPlaneLen = length(pPlane);
            const vec3 pRing = (pPlaneLen < 1e-10) ? vec3(0.0) : (p - (pPlane * r1 / pPlaneLen));

            return vec4(length(pRing) - r2, normalize(pRing));
        }

        __device__ __forceinline__ vec4 Box(const vec3& p, const float& size)
        {
            const float F = cwiseMax(abs(p));
            return vec4(F - size, floor(abs(p + vec3(1e-5) * sign(p)) / F) * sign(p));
        }

        __device__ __forceinline__ vec4 Sphere(const vec3& p, const float& r)
        {
            const float pLen = length(p);
            return vec4(pLen - r, vec3(p / pLen));
        }
    }
    
    // Foreward declarations
    namespace Host { class SDF; }

    enum __SDFPrimitive : int
    {    
        kSDFPrimitiveSphere,
        kSDFPrimitiveTorus,
        kSDFPrimitiveBox,
        kSDFPrimitiveCapsule,
        kNumSDFPrimitives
    };

    struct SDFParams
    {
        __host__ __device__ SDFParams();
        __host__ SDFParams(const ::Json::Node& node, const uint flags);

        __host__ void ToJson(::Json::Node& node) const;
        __host__ void FromJson(const ::Json::Node& node, const uint flags);
        __host__ void Update(const uint operation);

        int primitiveType;
        int maxSpecularIterations;
        int maxDiffuseIterations;
        float cutoffThreshold;
        float escapeThreshold;
        float rayIncrement;
        float failThreshold;
        float rayKickoff;

        struct { float r; } sphere;
        struct { float r1, r2; } torus;
        struct { float size; } box;

        TracableParams tracable;
        BidirectionalTransform transform;
    };

    namespace Device
    {       
        class SDF : public Device::Tracable
        {
            friend Host::SDF;

        protected:
            __device__ vec4 Field(vec3 p, const mat3& b, uint& code, uint& surfaceDepth) const;

            vec3        m_origin;

            SDFParams      m_params;

            uint        m_numVertices;
            uint        m_numFaces;
            uint        m_polyOrder;

        public:
            __device__ SDF();
            __device__ ~SDF() {}

            __device__ virtual bool Intersect(Ray& ray, HitCtx& hit) const override final;
            __device__ void Synchronise(const SDFParams& params)
            {
                m_params = params;
            }
        };
    }

    namespace Host
    {
        class SDF : public Host::Tracable
        {
        private:
            Device::SDF* cu_deviceData;
            SDFParams m_params;

        public:
            __host__ SDF(const std::string& id, const ::Json::Node& node);
            __host__ virtual ~SDF() = default;

            __host__ static AssetHandle<Host::RenderObject> Instantiate(const std::string& classId, const AssetType& expectedType, const ::Json::Node& json);

            __host__ virtual void OnDestroyAsset() override final;
            __host__ virtual void FromJson(const ::Json::Node& node, const uint flags) override final;
            __host__ static std::string GetAssetTypeString() { return "sdf"; }
            __host__ static std::string GetAssetDescriptionString() { return "SDF"; }
            __host__ virtual Device::SDF* GetDeviceInstance() const override final { return cu_deviceData; }
            __host__ virtual int GetIntersectionCostHeuristic() const override final { return 100; };
            __host__ virtual const RenderObjectParams* GetRenderObjectParams() const override final { return &m_params.tracable.renderObject; }
        };
    }
}