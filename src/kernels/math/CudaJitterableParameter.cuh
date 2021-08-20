#include "CudaMathUtils.cuh"

namespace Json { class Node; }

namespace Cuda
{
    template<typename PType>
    struct JitterableScalar
    {
        __device__ __host__ JitterableScalar() : p(0.0f), dpdt(0.0f), t(0.5f) {}
        __device__ __host__ JitterableScalar(const PType& v) : p(float(v)), dpdt(0.0f), t(0.5f) {}
        __host__ JitterableScalar(const std::string& id, const ::Json::Node& json, const uint flags) : JitterableScalar() { FromJson(id, json, flags);  }
        
        __host__ void FromJson(const std::string& id, const ::Json::Node& json, const uint flags);
        __host__ void ToJson(const std::string& id, ::Json::Node& json) const;
        __host__ void Randomise(float xi0, float xi1);
        __host__ inline PType Evaluate() const;

        __device__ __host__ JitterableScalar& operator=(const PType& other)
        {
            p = float(other);
            dpdt = float(0.0f);
            t = float(0.5f);
            return *this;
        }

    public:
        float p;
        float dpdt;
        float t;
    };

    template<typename PType, typename TType = PType>
    struct JitterableVec
    {
        __device__ __host__ JitterableVec() : p(0.0f), dpdt(0.0f), t(0.5f) {}
        __device__ __host__ JitterableVec(const PType& v) : p(TType(v)), dpdt(0.0f), t(0.5f) {}
        __host__ JitterableVec(const std::string& id, const ::Json::Node& json, const uint flags) : JitterableVec() { FromJson(id, json, flags); }

        __host__ void FromJson(const std::string& id, const ::Json::Node& json, const uint flags);
        __host__ void ToJson(const std::string& id, ::Json::Node& json) const;
        __host__ void Randomise(float xi0, float xi1);
        __host__  inline PType Evaluate() const;

        __device__ __host__  JitterableVec& operator=(const PType& other)
        {
            p = TType(other);
            dpdt = TType(0.0f);
            t = TType(0.5f);
            return *this;
        }

    public:
        TType p;
        TType dpdt;
        TType t;
    };

    template JitterableScalar<float>;
    template JitterableScalar<int>;
    template JitterableScalar<uint>;

    template JitterableVec<vec2>;
    template JitterableVec<vec3>;
    template JitterableVec<vec4>;
    template JitterableVec<ivec2, vec2>;
    template JitterableVec<ivec3, vec3>;
    template JitterableVec<ivec4, vec4>;

    using JitterableFloat = JitterableScalar<float>;
    using JitterableInt = JitterableScalar<int>;
    using JitterableUint = JitterableScalar<uint>;

    using JitterableVec2 = JitterableVec<vec2, vec2>;
    using JitterableVec3 = JitterableVec<vec3, vec3>;
    using JitterableVec4 = JitterableVec<vec4, vec4>;
    using JitterableIvec2 = JitterableVec<ivec2, vec2>;
    using JitterableIvec3 = JitterableVec<ivec3, vec3>;
    using JitterableIvec4 = JitterableVec<ivec4, vec4>;
}