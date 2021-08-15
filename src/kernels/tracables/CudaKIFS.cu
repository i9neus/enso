#include "CudaKIFS.cuh"
#include "CudaSDF.cuh"
#include "generic/JsonUtils.h"

namespace Cuda
{
#define kKIFSPolyhedronType 0
#define kKIFSFoldType       0

    /*constexpr uint kSDFMaxSteps = 50;
    constexpr float kSDFCutoffThreshold = 1e-4f;
    constexpr float kSDFEscapeThreshold = 1.0f;
    constexpr float kSDFRayIncrement = 0.9f;
    constexpr float kSDFFailThreshold = 1e-2f;*/

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
        faceMask(0xffffffff),
        foldType(kKIFSTetrahedtron),
        primitiveType(kKIFSTetrahedtron)
    {
        sdf.maxSpecularIterations = 50;
        sdf.maxDiffuseIterations = 15;
        sdf.cutoffThreshold = 1e-4f;
        sdf.escapeThreshold = 1.0f;
        sdf.rayIncrement = 0.9f;
        sdf.rayKickoff = 1e-4f;
        sdf.failThreshold = 1e-2f;
        sdf.clipCameraRays = true;
        sdf.clipShape = kKIFSBox;
    }

    __host__ KIFSParams::KIFSParams(const ::Json::Node& node, const uint flags) :
        KIFSParams()
    { 
        FromJson(node, flags); 
    }

    __host__ void KIFSParams::ToJson(::Json::Node& node) const
    {
        node.AddArray("rotate", std::vector<float>({ rotate.x, rotate.y}));
        node.AddArray("scale", std::vector<float>({ scale.x, scale.y }));
        node.AddValue("vertScale", vertScale);
        node.AddValue("crustThickness", crustThickness);
        node.AddValue("numIterations", numIterations);
        node.AddValue("faceMask", faceMask);
        node.AddEnumeratedParameter("foldType", std::vector<std::string>({ "tetrahedron", "cube" }), foldType);
        node.AddEnumeratedParameter("primitiveType", std::vector<std::string>({ "tetrahedron", "cube" }), primitiveType); 

        ::Json::Node sdfNode = node.AddChildObject("sdf");
        sdfNode.AddValue("maxSpecularIterations", sdf.maxSpecularIterations);
        sdfNode.AddValue("maxDiffuseIterations", sdf.maxDiffuseIterations);
        sdfNode.AddValue("cutoffThreshold", sdf.cutoffThreshold);
        sdfNode.AddValue("escapeThreshold", sdf.escapeThreshold);
        sdfNode.AddValue("rayIncrement", sdf.rayIncrement);
        sdfNode.AddValue("rayKickoff", sdf.rayKickoff);
        sdfNode.AddValue("failThreshold", sdf.failThreshold);
        sdfNode.AddValue("clipCameraRays", sdf.clipCameraRays);
        sdfNode.AddEnumeratedParameter("clipShape", std::vector<std::string>({ "box", "sphere", "torus" }), sdf.clipShape);

        transform.ToJson(node);
    }

    __host__ void KIFSParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        node.GetVector("rotate", rotate, flags);
        node.GetVector("scale", scale, flags);
        node.GetValue("vertScale", vertScale, flags);
        node.GetValue("crustThickness", crustThickness, flags);
        node.GetValue("numIterations", numIterations, flags);
        node.GetValue("faceMask", faceMask, flags);
        node.GetEnumeratedParameter("foldType", std::vector<std::string>({"tetrahedron", "cube" }), foldType, flags);
        node.GetEnumeratedParameter("primitiveType", std::vector<std::string>({ "tetrahedron", "cube" }), primitiveType, flags);

        const ::Json::Node sdfNode = node.GetChildObject("sdf", flags);
        if (sdfNode)
        {
            sdfNode.GetValue("maxSpecularIterations", sdf.maxSpecularIterations, flags);
            sdfNode.GetValue("maxDiffuseIterations", sdf.maxDiffuseIterations, flags);
            sdfNode.GetValue("cutoffThreshold", sdf.cutoffThreshold, flags);
            sdfNode.GetValue("escapeThreshold", sdf.escapeThreshold, flags);
            sdfNode.GetValue("rayIncrement", sdf.rayIncrement, flags);
            sdfNode.GetValue("rayKickoff", sdf.rayKickoff, flags);
            sdfNode.GetValue("failThreshold", sdf.failThreshold, flags);
            sdfNode.GetValue("clipCameraRays", sdf.clipCameraRays, flags);

            sdfNode.GetEnumeratedParameter("clipShape", std::vector<std::string>({ "box", "sphere", "torus" }), sdf.clipShape, flags);
        }

        transform.FromJson(node, flags);
    }

    __host__ bool KIFSParams::operator==(const KIFSParams& rhs) const
    {
        return rotate == rhs.rotate &&
            scale == rhs.scale &&
            vertScale == rhs.vertScale &&
            crustThickness == rhs.crustThickness &&
            numIterations == rhs.numIterations &&
            faceMask == rhs.faceMask;
    }

    __device__ void Device::KIFS::Prepare()
    {
        auto& kcd = m_kernelConstantData;
        kcd.numIterations = m_params.numIterations;
        kcd.vertScale = powf(2.0, mix(-8.0f, 8.0f, m_params.vertScale));
        kcd.crustThickness = powf(2.0f, mix(-15.0f, 0.0f, m_params.crustThickness));
        kcd.faceMask = m_params.faceMask;
        kcd.foldType = m_params.foldType;
        kcd.primitiveType = m_params.primitiveType;

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

            kcd.iteration[i].matrix = BidirectionalTransform(vec3(0.0f), vec3(theta), vec3(iterScale)).fwd;
            kcd.iteration[i].scale = iterScale;
        }
    }

    __device__ Device::KIFS::KIFS()
    {
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

    __device__ __forceinline__ void FoldTetrahedron(const int& i, vec3& p, mat3& bi, ushort& code)
    {
        if (p.x + p.y < 0)
        {
            p.xy = -p.yx;
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

    __device__ __forceinline__ void FoldCube(const int& i, vec3& p, mat3& bi, ushort& code)
    {
        for (int d = 0; d < 3; d++)
        {
            if (p[d] < 0.0)
            {
                p[d] = -p[d];
                bi[0][d] = -bi[0][d];
                bi[1][d] = -bi[1][d];
                bi[2][d] = -bi[2][d];
                code |= (1u << (3 * i + d));
            }
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
    __device__ __forceinline__ vec4 SDFPolyhedronFace(const vec3& p, const PolyType& poly, const uint& faceIdx, const float& scale)
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

    template<typename PolyType>
     __device__ __forceinline__ void SDFPolyhedron(vec4& F, const Device::KIFS::KernelConstantData& kcd, const float& iterationScale, const vec3& p, const PolyType& poly)
    {
         for (int i = 0; i < poly.kNumFaces; i++)
         {
             if ((kcd.faceMask & (1 << i)) != 0)
             {
                 vec4 FFace = SDFPolyhedronFace(p, poly, i, kcd.vertScale);

                 FFace.x = (FFace.x * iterationScale) - kcd.crustThickness;
                 if (FFace.x < F.x) { F = FFace; }
             }
         }
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

            if (kcd.foldType == kKIFSTetrahedtron)
            {
                FoldTetrahedron(i, p, bi, code);
            }
            else
            {
                FoldCube(i, p, bi, code);
            }

            // Scale up around [1, 1, 1]
            p = (p - kOne) * 2.0f + kOne;
            iterationScale *= 0.5f;
        }    

        // Test this position against each polyhedron face
        F.x = kFltMax;
        if (kcd.primitiveType == kKIFSTetrahedtron)
        {
            SDFPolyhedron(F, kcd, iterationScale, p, kTetrahedronData);
        }  
        else
        {
            SDFPolyhedron(F, kcd, iterationScale, p, kCubeData);
        }

        // Transform the normal from folded space into object space
        F.yzw = bi * F.yzw;
        F.yzw = normalize(basis[0] * F.y + basis[1] * F.z + basis[2] * F.w);
        pathId = code;

        return F;
    }

    __device__  bool Device::KIFS::Intersect(Ray& globalRay, HitCtx& hitCtx) const
    {
        RayBasic localRay = RayToObjectSpace(globalRay.od, m_params.transform);

        float t = Intersector::RayBox(localRay, 1.0f);
        if (t == kNoIntersect) { return false; }

        //float t = 0.0f; 

        const float localMag = length(localRay.d);
        localRay.d /= localMag;
        t *= localMag;

        const mat3 basis = CreateBasis(localRay.d);
        vec3 grad;
        vec3 p = localRay.PointAt(t);
        int i;
        vec4 F;
        bool isSubsurface = false;
        uint code = 0;
        uint surfaceDepth = 0;
        bool hitBoundSDF = false;
        const int maxIterations = (!(globalRay.flags & kRayScattered)) ? m_params.sdf.maxSpecularIterations : m_params.sdf.maxDiffuseIterations;

        for (i = 0; i < maxIterations; i++)
        {
            F = Field(p, basis, code, surfaceDepth);
            hitBoundSDF = false;

            if(m_params.sdf.clipCameraRays && (globalRay.depth == 1 || (globalRay.depth == 2 && globalRay.flags & kRayDirectSample)))
            {
                vec3 pWorld = globalRay.PointAt(t / localMag);
                vec4 FClip;
                switch (m_params.sdf.clipShape)
                {
                case kKIFSSphere:
                    FClip = SDF::Sphere(pWorld, 0.5f); break;
                case kKIFSTorus:
                    FClip = SDF::Torus(pWorld, 0.4f, 0.1f); break;
                default:
                    FClip = SDF::Box(pWorld, 0.5f);
                }
                FClip.x *= localMag;
                if (FClip.x > F.x) 
                { 
                    F = FClip; 
                    hitBoundSDF = true;
                }
            }

            // On the first iteration, simply determine whether we're inside the isosurface or not
            if (i == 0) { isSubsurface = F.x < 0.0; }
            // Otherwise, check to see if we're at the surface
            else if (F.x > 0.0 && F.x < m_params.sdf.cutoffThreshold) { hitCtx.debug = vec3(0.0f, 0.0f, 1.0f) * float(i) / float(maxIterations); break; }

            if (F.x > m_params.sdf.escapeThreshold) { hitCtx.debug = vec3(1.0, 0.0f, 0.0f) * float(i) / float(maxIterations); return false; }

            // Increment the ray position based on the SDF magnitude
            t += (isSubsurface ? -F.x : F.x) * m_params.sdf.rayIncrement;            
            
            // If we've gone beyond t-near, bail out now
            if (t / localMag > globalRay.tNear) { hitCtx.debug = vec3(1.0, 1.0f, 0.0f) * float(i) / float(maxIterations); return false; }

            p = localRay.PointAt(t);
        }

        if (F.x > m_params.sdf.failThreshold) { hitCtx.debug = vec3(0.0f, 1.0f, 0.0f) * float(i) / float(maxIterations); return false; }
        t /= localMag;
        hitCtx.debug *= F.x;

        globalRay.tNear = t;
        hitCtx.Set(HitPoint(globalRay.HitPoint(), 
                    hitBoundSDF ? F.yzw : NormalToWorldSpace(F.yzw, m_params.transform)),
                    false, 
                    vec2(*reinterpret_cast<float*>(&code), 0.0f),  // Dump the bits of the code into the float. FIXME: Not type safe, so fix this
                    m_params.sdf.rayKickoff);

        return true;
    }

    __host__ AssetHandle<Host::RenderObject> Host::KIFS::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kTracable) { return AssetHandle<Host::RenderObject>(); }

        return AssetHandle<Host::RenderObject>(new Host::KIFS(json), id);
    }

    __host__  Host::KIFS::KIFS(const ::Json::Node& node)
        : cu_deviceData(nullptr)
    {
        cu_deviceData = InstantiateOnDevice<Device::KIFS>();
        FromJson(node, ::Json::kRequiredWarn);

        {
            SimplePolyhedron<4, 4, 3> tet = {
                { vec3(1.0f, 1.0f, 1.0f) * 0.5f, vec3(-1.0f, -1.0f, 1.0f) * 0.5f, vec3(1.0f, -1.0f, -1.0f) * 0.5f, vec3(-1.0f, 1.0f, -1.0f) * 0.5f },
                { 0, 1, 2, 1, 2, 3, 2, 3, 0, 3, 0, 1 }
            };
            tet.sqrBoundRadius = length2(vec3(1.0f, 1.0f, 1.0f) * 0.5f) + 1e-6f;
            IsOk(cudaMemcpyToSymbol(kTetrahedronData, &tet, sizeof(SimplePolyhedron<4, 4, 3>)));
        }

        {
            SimplePolyhedron<8, 6, 4> tet = {
               { vec3(1.0f, 1.0f, 1.0f) * 0.5f, vec3(-1.0f, 1.0f, 1.0f) * 0.5f, vec3(1.0f, -1.0f, 1.0f) * 0.5f, vec3(-1.0f, -1.0f, 1.0f) * 0.5f,
                 vec3(1.0f, 1.0f, -1.0f) * 0.5f, vec3(-1.0f, 1.0f, -1.0f) * 0.5f, vec3(1.0f, -1.0f, -1.0f) * 0.5f, vec3(-1.0f, -1.0f, -1.0f) * 0.5f },
               { 0, 1, 3, 2, 4, 5, 7, 6, 0, 1, 5, 4, 2, 3, 7, 6, 0, 2, 6, 4, 1, 3, 7, 5 }
            };
            tet.sqrBoundRadius = length2(vec3(1.0f, 1.0f, 1.0f) * 0.5f) + 1e-6f;
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
        DestroyOnDevice(cu_deviceData);
    }

    __host__ void Host::KIFS::FromJson(const ::Json::Node& node, const uint flags)
    {
        Host::Tracable::FromJson(node, flags);
        
        SynchroniseObjects(cu_deviceData, KIFSParams(node, flags));
    }
}