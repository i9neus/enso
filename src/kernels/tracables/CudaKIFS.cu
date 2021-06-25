#include "CudaKIFS.cuh"
#include "CudaSDF.cuh"
#include "generic/JsonUtils.h"

namespace Cuda
{
#define kKIFSPolyhedronType 0
#define kKIFSFoldType       0

    constexpr uint kSDFMaxSteps = 50;
    constexpr float kSDFCutoffThreshold = 1e-4f;
    constexpr float kSDFEscapeThreshold = 1.0f;
    constexpr float kSDFRayIncrement = 0.9f;
    constexpr float kSDFFailThreshold = 1e-2f;

    __constant__ SimplePolyhedron<4, 4, 3> kTetrahedronData;
    __constant__ SimplePolyhedron<8, 6, 4> kCubeData;
    __constant__ vec3 kXi[kSDFMaxIterations];

    struct ThreadData
    {
        __device__ ThreadData() {}

        vec3    p[16 * 16];
        union
        {
            // Folding/rotating 
            struct
            {
                ushort  code[16 * 16];
                mat3    bi[16 * 16];
                float   iterationScale[16 * 16];
                vec4    F[16 * 16];
                int     i[16 * 16];
            };
        };
    };

    __host__ __device__ KIFSParams::KIFSParams() :
        rotate(0.0f),
        scale(0.5f),
        vertScale(0.5f),
        crustThickness(0.5f),
        numIterations(1),
        faceMask(0xffffffff)
    {
    }

    __host__ void KIFSParams::ToJson(Json::Node& node) const
    {
        node.AddArray("rotate", std::vector<float>({ rotate.x, rotate.y, rotate.z }));
        node.AddArray("scale", std::vector<float>({ scale.x, scale.y }));
        node.AddValue("vertScale", vertScale);
        node.AddValue("crustThickness", crustThickness);
        node.AddValue("numIterations", numIterations);
        node.AddValue("faceMask", faceMask);
    }

    __host__ void KIFSParams::FromJson(const Json::Node& node)
    {
        node.GetVector("rotate", rotate, true);
        node.GetVector("scale", scale, true);
        node.GetValue("vertScale", vertScale, true);
        node.GetValue("crustThickness", crustThickness, true);
        node.GetValue("numIterations", numIterations, true);
        node.GetValue("faceMask", faceMask, true);
    }

    __device__ void Device::KIFS::Prepare()
    {
        m_kernelConstantData.numIterations = m_params.numIterations;
        m_kernelConstantData.vertScale = powf(2.0, mix(-8.0f, 8.0f, m_params.vertScale));
        m_kernelConstantData.crustThickness = powf(2.0f, mix(-15.0f, 0.0f, m_params.crustThickness));
        m_kernelConstantData.faceMask = m_params.faceMask;

        float rotateAlpha = mix(-1.0f, 1.0f, m_params.rotate.x);
        rotateAlpha = 2.0 * powf(fabsf(rotateAlpha), 2.0f) * sign(rotateAlpha);
        float rotateBeta = mix(-1.0f, 1.0f, m_params.rotate.y);
        rotateBeta = 2.0 * powf(fabsf(rotateBeta), 2.0f) * sign(rotateBeta);

        float scaleAlpha = mix(0.5, 1.5, m_params.scale.x);
        float scaleBeta = mix(-1.0, 1.0, m_params.scale.y);

        for (int i = 0; i < kSDFMaxIterations; i++)
        {
            float f = float(i + 1) / float(max(1, kSDFMaxIterations));
            vec3 theta = (2.0f * kXi[i] - 1.0f) * kPi * (rotateAlpha + rotateBeta * f);
            float iterScale = scaleAlpha * powf(2.0f, float(i) * scaleBeta);

            m_kernelConstantData.iteration[i].matrix = BidirectionalTransform(vec3(0.0f), vec3(theta), vec3(iterScale)).fwd;
            m_kernelConstantData.iteration[i].scale = iterScale;
        }
    }

    __device__ Device::KIFS::KIFS()
    {
        m_type = KIFSType::kTetrahedtron;

        Prepare();
    }

    __device__ Device::KIFS::KernelConstantData& GetKernelConstantData()
    {
        __device__ __shared__ Device::KIFS::KernelConstantData kcd;
        return kcd;
    }

    __device__ void Device::KIFS::InitialiseKernelConstantData() const
    {
        memcpy(&GetKernelConstantData(), &m_kernelConstantData, sizeof(KernelConstantData));
    }

    __device__ __forceinline__ void FoldTetrahedron(const mat3& matrix, const int& i, vec3& p, mat3& bi, ushort& code)
    {
        if (p.x + p.y < 0)
        {
            p.xy = -p.xy;
            bi[0].xy = -bi[0].yx;
            bi[1].xy = -bi[1].yx;
            bi[2].xy = -bi[2].yx;
            code |= (1u << (3 * i));
        }
        if (p.x + p.z < 0)
        {
            p.xz = -p.zx;
            bi[0].xz = -bi[0].zx;
            bi[1].xz = -bi[1].zx;
            bi[2].xz = -bi[2].zx;
            code |= (1u << (3 * i + 1));
        }
        if (p.y + p.z < 0)
        {
            p.zy = -p.yz;
            bi[0].zy = -bi[0].yz;
            bi[1].zy = -bi[1].yz;
            bi[2].zy = -bi[2].yz;
            code |= (1u << (3 * i + 2));
        }
    }

    __device__ uint Device::KIFS::SizeOfSharedMemory()
    {
        uint bytes = 0;
        bytes += sizeof(KernelConstantData);
        bytes += sizeof(ThreadData);
        return bytes;
    }

    template<typename PolyType>
    __device__ __forceinline__ vec4 PolyhedronFace(const vec3& p, const PolyType& poly, const uint& faceIdx, const float& scale)
    {
        const vec3* V = poly.V;
        const uchar* F = &poly.F[faceIdx * PolyType::kPolyOrder];
        const vec3 N = normalize(cross(V[F[1]] - V[F[0]], V[F[2]] - V[F[0]]));
        vec3 grad;

        // Test against each of the polygon's edges
        for (int i = 0; i < PolyType::kPolyOrder; i++)
        {
            const vec3 dv = (V[F[(i + 1) % poly.kPolyOrder]] - V[F[i]]) * scale;
            const vec3 vi = V[F[i]] * scale;
            const vec3 edgeNorm = normalize(cross(dv, N));
            if (dot(edgeNorm, p - vi) > 0.0f)
            {
                const float t = clamp((dot(p, dv) - dot(vi, dv)) / dot(dv, dv), 0.0f, 1.0f);
                grad = p - (vi + t * dv);
                const float gradMag = length(grad);
                return vec4(gradMag, grad / gradMag);
            }
        }

        // Test against the face itself
        const vec3 v0 = V[F[0]] * scale;
        return (dot(N, p - v0) < 0.0f) ? vec4((dot(p, -N) - dot(v0, -N)), -N) : vec4((dot(p, N) - dot(v0, N)), N);
    }

    __device__ vec4 Device::KIFS::Field(vec3 position, const mat3& basis, uint& pathId, uint& surfaceDepth) const
    {
        const auto& kcd = GetKernelConstantData();

        __shared__  ThreadData td;
        auto& p = td.p[kThreadIdx];
        auto& bi = td.bi[kThreadIdx];
        auto& iterationScale = td.iterationScale[kThreadIdx];
        auto& code = td.code[kThreadIdx];
        auto& F = td.F[kThreadIdx];
        auto& i = td.i[kThreadIdx];

        p = position;
        bi = basis;
        iterationScale = 1.0f;
        code = 0u;

        for (i = 0; i < kcd.numIterations; i++)
        {
            const mat3& matrix = kcd.iteration[i].matrix;
            const float& iterScale = kcd.iteration[i].scale;

            p = matrix * p;
            bi[0] = matrix * bi[0];
            bi[1] = matrix * bi[1];
            bi[2] = matrix * bi[2];
            iterationScale *= iterScale;

            if (i == kcd.numIterations - 1) { break; }

#if kKIFSFoldType == 0
            FoldTetrahedron(matrix, i, p, bi, code);
#else
            FoldCube(matrix, i, p, bi, code);
#endif

            p = (p - kOne) * 2.0f + kOne;
            iterationScale *= 0.5f;
        }

        // Transform the normal from folded space into object space
        F.yzw = bi * F.yzw;
        F.yzw = normalize(basis[0] * F.y + basis[1] * F.z + basis[2] * F.w);
        pathId = code;

        // Test this position against each polyhedron face
        F.x = kFltMax;
        for (i = 0; i < kCubeData.kNumFaces; i++)
        {
            //if ((kcd.faceMask & (1 << i)) != 0)
            {
                vec4 FFace = PolyhedronFace(p, kTetrahedronData, i, kcd.vertScale);

                FFace.x = (FFace.x * iterationScale) - kcd.crustThickness;
                if (FFace.x < F.x) { F = FFace; }
            }
        }

        //F= SDF::Torus(position, 0.3, 0.1);

        return F;
    }

    __device__  bool Device::KIFS::Intersect(Ray& ray, HitCtx& hitCtx) const
    {
        RayBasic localRay = RayToObjectSpace(ray.od, m_transform);

        float t = Intersector::RayUnitBox(localRay);
        if (t == kNoIntersect) { return false; }
        //float t = 0.0f;

        const float transformScale = 1.0f;
        const float localMag = length(localRay.d);
        localRay.d /= localMag;

        const mat3 basis = CreateBasis(localRay.d);
        vec3 grad;
        vec3 p = localRay.PointAt(t);
        int i;
        vec4 F;
        bool isSubsurface = false;
        uint code = 0;
        uint surfaceDepth = 0;

        for (i = 0; i < kSDFMaxSteps; i++)
        {
            F = Field(p, basis, code, surfaceDepth);

            //if(!isOriginInBound)
            {
                vec4 FBox = SDF::Box(p, 0.5f);
                //vec4 FBox = SDF::Torus(p, 0.25 / transformScale, 0.1 / transformScale);
                //vec4 FBox = SDF::Sphere(p, 0.5 / transformScale);
                if (FBox.x > F.x) { F = FBox; }
                //F = FBox;
            }

            // On the first iteration, simply determine whether we're inside the isosurface or not
            if (i == 0) { isSubsurface = F.x < 0.0; }
            // Otherwise, check to see if we're at the surface
            else if (F.x > 0.0 && F.x < kSDFCutoffThreshold) { break; }

            if (F.x > kSDFEscapeThreshold) { return false; }

            t += isSubsurface ? -F.x : F.x;
            p = localRay.PointAt(t);
        }

        t /= localMag;
        if (F.x > kSDFFailThreshold || t > ray.tNear) { return false; }

        ray.tNear = t;
        hitCtx.Set(HitPoint(ray.HitPoint(), NormalToWorldSpace(F.yzw, m_transform)), false, vec2(0.0f), 1e-4f);

        return true;
    }

    __host__  Host::KIFS::KIFS(const Json::Node& node)
        : cu_deviceData(nullptr)
    {
        cu_deviceData = InstantiateOnDevice<Device::KIFS>();
        FromJson(node);

        {
            SimplePolyhedron<4, 4, 3> tet = {
                { vec3(1.0f, 1.0f, 1.0f) * 0.5f, vec3(-1.0f, -1.0f, 1.0f) * 0.5f, vec3(1.0f, -1.0f, -1.0f) * 0.5f, vec3(-1.0f, 1.0f, -1.0f) * 0.5f },
                { 0, 1, 2, 1, 2, 3, 2, 3, 0, 3, 0, 1 }
            };
            IsOk(cudaMemcpyToSymbol(kTetrahedronData, &tet, sizeof(SimplePolyhedron<4, 4, 3>)));
        }

        {
            SimplePolyhedron<8, 6, 4> tet = {
               { vec3(1.0f, 1.0f, 1.0f) * 0.5f, vec3(-1.0f, 1.0f, 1.0f) * 0.5f, vec3(1.0f, -1.0f, 1.0f) * 0.5f, vec3(-1.0f, -1.0f, 1.0f) * 0.5f,
                 vec3(1.0f, 1.0f, -1.0f) * 0.5f, vec3(-1.0f, 1.0f, -1.0f) * 0.5f, vec3(1.0f, -1.0f, -1.0f) * 0.5f, vec3(-1.0f, -1.0f, -1.0f) * 0.5f },
               { 0, 1, 3, 2, 4, 5, 7, 6, 0, 1, 5, 4, 2, 3, 7, 6, 0, 2, 6, 4, 1, 3, 7, 5 }
            };
            IsOk(cudaMemcpyToSymbol(kCubeData, &tet, sizeof(SimplePolyhedron<8, 6, 4>)));
        }

        {
            vec3 xi[kSDFMaxIterations] = {
                vec3(0.86418, 0.550822, 0.123257), vec3(0.39088, 0.450768, 0.883838), vec3(0.99579, 0.614493, 0.0399468), vec3(0.022739, 0.691733, 0.23853),
                vec3(0.0491209, 0.14045, 0.0436547), vec3(0.566151, 0.644241, 0.559143),  vec3(0.336939, 0.0539437, 0.244569), vec3(0.240348, 0.349121, 0.391118),
                vec3(0.0955733, 0.642528, 0.46847), vec3(0.188407, 0.360586, 0.659837)
            };
            IsOk(cudaMemcpyToSymbol(kXi, xi, sizeof(vec3) * kSDFMaxIterations));
        }
    }

    __host__ void Host::KIFS::OnDestroyAsset()
    {
        DestroyOnDevice(&cu_deviceData);
    }

    __host__ void Host::KIFS::OnJson(const Json::Node& parentNode)
    {
        Json::Node childNode = parentNode.GetChildObject("kifs", true);
        if (childNode)
        {
            SyncParameters(cu_deviceData, KIFSParams(childNode));
        }
    }
}