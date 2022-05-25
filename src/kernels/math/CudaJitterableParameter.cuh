#include "CudaMathUtils.cuh"

namespace Json { class Node; }

namespace Cuda
{
    enum JitterOperation : int
    {
        kJitterRandomise,
        kJitterReset,
        kJitterFlatten
    };
    
    template<typename PType>
    struct JitterableScalar
    {
        __device__ __host__ JitterableScalar() : eval(0.0f), p(0.0f), dpdt(0.0f), t(0.5f) {}
        __device__ __host__ JitterableScalar(const PType& v) : eval(v), p(float(v)), dpdt(0.0f), t(0.5f) {}
        __host__ JitterableScalar(const std::string& id, const ::Json::Node& json, const uint flags) : JitterableScalar() { FromJson(id, json, flags);  }

        __device__ __host__ __forceinline__ const PType& operator()(void) const { return eval; }
        
        __host__ uint FromJson(const std::string& id, const ::Json::Node& json, const uint flags);
        __host__ void ToJson(const std::string& id, ::Json::Node& json) const;
        __host__ void Update(const int operation);
        __host__ inline void Evaluate();

        __device__ __host__ JitterableScalar& operator=(const PType& other)
        {
            eval = other;
            p = float(other);
            dpdt = float(0.0f);
            t = float(0.5f);

            return *this;
        }

    public:
        PType eval;
        float p;
        float dpdt;
        float t;
    };

    template<typename PType, typename TType = PType>
    struct JitterableVec
    {
        __device__ __host__ JitterableVec() : eval(PType::kType(0)), p(TType::kType(0)), dpdt(TType::kType(0)), t(0.5f) {}
        __device__ __host__ JitterableVec(const PType& v) : eval(TType(v)), p(TType(v)), dpdt(TType::kType(0)), t(TType::kType(0.5)) {}
        __host__ JitterableVec(const std::string& id, const ::Json::Node& json, const uint flags) : JitterableVec() { FromJson(id, json, flags); }

        __device__ __host__ __forceinline__ const PType& operator()(void) const { return eval; }

        __host__ uint FromJson(const std::string& id, const ::Json::Node& json, const uint flags);
        __host__ void ToJson(const std::string& id, ::Json::Node& json) const;
        __host__ void Update(const int operation);
        __host__  inline void Evaluate();

        __device__ __host__  JitterableVec& operator=(const PType& other)
        {
            eval = other;
            p = TType(other);
            dpdt = TType(0.0f);
            t = TType(0.5f);

            return *this;
        }

    public:
        PType eval;
        TType p;
        TType dpdt;
        TType t;
    };

    struct JitterableFlags
    {
        __device__ __host__ JitterableFlags() : eval(0), p(0), dpdt(0), t(0), validBits(0) {}
        __device__ __host__ JitterableFlags(const uint& v, const uchar bits) : eval(v), p(v), dpdt(0), t(0), validBits(bits) {}
        __host__ JitterableFlags(const std::string& id, const uchar bits, const ::Json::Node& json, const uint flags) : JitterableFlags() 
        { 
            validBits = bits;
            FromJson(id, json, flags); 
        }

        __device__ __host__ __forceinline__ const uint& operator()(void) const { return eval; }
        __device__ __host__ __forceinline__ const bool operator()(const uint bit) const { return (eval >> bit) & 1; }

        __host__ uint FromJson(const std::string& id, const ::Json::Node& json, const uint flags);
        __host__ void ToJson(const std::string& id, ::Json::Node& json) const;
        __host__ void Update(const int operation);
        __host__  inline void Evaluate();

        __device__ __host__  JitterableFlags& operator=(const uint& other)
        {
            eval = other;
            p = other;
            dpdt = 0;
            t = 0;

            return *this;
        }

    public:
        uint eval;
        uint p;
        uint dpdt;
        uint t;
        uchar validBits;
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