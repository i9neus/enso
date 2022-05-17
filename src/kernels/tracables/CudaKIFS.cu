#include "CudaKIFS.cuh"
#include "CudaSDF.cuh"
#include "generic/JsonUtils.h"

#include <random>

namespace Cuda
{
#define kKIFSPolyhedronType 0
#define kKIFSFoldType       0

    /*constexpr uint kSDFMaxSteps = 50;
    constexpr float kSDFCutoffThreshold = 1e-4f;
    constexpr float kSDFEscapeThreshold = 1.0f;
    constexpr float kSDFRayIncrement = 0.9f;
    constexpr float kSDFFailThreshold = 1e-2f;*/

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
        rotateA(0.5f),
        rotateB(0.5f),
        scaleA(0.5f),
        scaleB(0.5f),
        vertScale(0.5f),
        crustThickness(0.5f),        
        numIterations(1),
        faceMask(0xffffffff, 6),
        foldType(kKIFSFoldTetrahedron),
        primitiveType(kKIFSPrimitiveTetrahedronSolid),
        doTakeSnapshot(false)
    {        
        sdf.maxSpecularIterations = 50;
        sdf.maxDiffuseIterations = 15;
        sdf.cutoffThreshold = 1e-5f;
        sdf.escapeThreshold = 1.0f;
        sdf.rayIncrement = 0.9f;
        sdf.rayKickoff = 1e-4f;
        sdf.failThreshold = 1e-2f;
        sdf.flags = JitterableFlags(kKIFSClipRays, 1),
        sdf.clipShape = kKIFSClipBox;
    }

    __host__ KIFSParams::KIFSParams(const ::Json::Node& node, const uint flags) :
        KIFSParams()
    { 
        FromJson(node, flags); 
    }

    __host__ void KIFSParams::Update(const uint operation)
    {
        rotateA.Update(operation);
        rotateB.Update(operation);
        scaleA.Update(operation);
        scaleB.Update(operation);
        vertScale.Update(operation);
        crustThickness.Update(operation);
        faceMask.Update(operation);
        sdf.flags.Update(operation);

        transform.Update(operation);
    }

    __host__ void KIFSParams::ToJson(::Json::Node& node) const
    {
        rotateA.ToJson("rotateA", node);
        rotateB.ToJson("rotateB", node);
        scaleA.ToJson("scaleA", node);
        scaleB.ToJson("scaleB", node);
        crustThickness.ToJson("crustThickness", node);
        vertScale.ToJson("vertScale", node);
        faceMask.ToJson("faceMask", node);

        node.AddValue("numIterations", numIterations);
        node.AddEnumeratedParameter("foldType", std::vector<std::string>({ "tetrahedron", "cube" }), foldType);
        node.AddEnumeratedParameter("primitiveType", std::vector<std::string>({ "tetrahedron", "cube", "sphere", "torus", "box", "tetrahedroncage", "cubecage" }), primitiveType); 

        ::Json::Node sdfNode = node.AddChildObject("sdf");
        sdf.flags.ToJson("flags", sdfNode);
        sdfNode.AddValue("maxSpecularIterations", sdf.maxSpecularIterations);
        sdfNode.AddValue("maxDiffuseIterations", sdf.maxDiffuseIterations);
        sdfNode.AddValue("cutoffThreshold", sdf.cutoffThreshold);
        sdfNode.AddValue("escapeThreshold", sdf.escapeThreshold);
        sdfNode.AddValue("rayIncrement", sdf.rayIncrement);
        sdfNode.AddValue("rayKickoff", sdf.rayKickoff);
        sdfNode.AddValue("failThreshold", sdf.failThreshold);
        sdfNode.AddEnumeratedParameter("clipShape", std::vector<std::string>({ "box", "sphere", "torus" }), sdf.clipShape);

        tracable.ToJson(node);
        transform.ToJson(node);
    }

    __host__ void KIFSParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        rotateA.FromJson("rotateA", node, flags);
        rotateB.FromJson("rotateB", node, flags);
        scaleA.FromJson("scaleA", node, flags);
        scaleB.FromJson("scaleB", node, flags);
        vertScale.FromJson("vertScale", node, flags);
        crustThickness.FromJson("crustThickness", node, flags);
        faceMask.FromJson("faceMask", node, flags);

        node.GetValue("numIterations", numIterations, flags);
        node.GetEnumeratedParameter("foldType", std::vector<std::string>({"tetrahedron", "cube" }), foldType, flags);
        node.GetEnumeratedParameter("primitiveType", std::vector<std::string>({ "tetrahedron", "cube", "sphere", "torus", "box", "tetrahedroncage", "cubecage" }), primitiveType, flags);

        const ::Json::Node sdfNode = node.GetChildObject("sdf", flags);
        if (sdfNode)
        {
            sdf.flags.FromJson("flags", sdfNode, flags);
            sdfNode.GetValue("maxSpecularIterations", sdf.maxSpecularIterations, flags);
            sdfNode.GetValue("maxDiffuseIterations", sdf.maxDiffuseIterations, flags);
            sdfNode.GetValue("cutoffThreshold", sdf.cutoffThreshold, flags);
            sdfNode.GetValue("escapeThreshold", sdf.escapeThreshold, flags);
            sdfNode.GetValue("rayIncrement", sdf.rayIncrement, flags);
            sdfNode.GetValue("rayKickoff", sdf.rayKickoff, flags);
            sdfNode.GetValue("failThreshold", sdf.failThreshold, flags);

            sdfNode.GetEnumeratedParameter("clipShape", std::vector<std::string>({ "box", "sphere", "torus" }), sdf.clipShape, flags);
        }

        tracable.FromJson(node, flags);
        transform.FromJson(node, flags);
    }

    __device__ void Device::KIFS::Prepare()
    {
        auto& kcd = m_kernelConstantData;
        kcd.numIterations = m_params.numIterations;
        kcd.vertScale = powf(2.0, mix(-8.0f, 8.0f, m_params.vertScale()));
        kcd.crustThickness = powf(2.0f, mix(-15.0f, 0.0f, m_params.crustThickness()));
        kcd.faceMask = m_params.faceMask();
        kcd.foldType = m_params.foldType;
        kcd.primitiveType = m_params.primitiveType;

        // Precompute the tetrahedron data
        {
            auto& tet = kcd.tetrahedronData;
            tet.V[0] = vec3(1.0f, 1.0f, 1.0f);
            tet.V[1] = vec3(-1.0f, -1.0f, 1.0f);
            tet.V[2] = vec3(1.0f, -1.0f, -1.0f);
            tet.V[3] = vec3(-1.0f, 1.0f, -1.0f);

            const uchar F[] = { 0, 1, 2, 1, 2, 3, 2, 3, 0, 3, 0, 1 };
            memcpy(tet.F, F, sizeof(uchar) * 12);
            const uchar E[] = { 0, 1, 1, 2, 2, 0, 0, 3, 1, 3, 2, 3 };
            memcpy(tet.E, E, sizeof(uchar) * 12);

            tet.Prepare(kcd.vertScale * 0.5f);
        }

        // Precompute the cube data
        {
            auto& cube = kcd.cubeData;
            cube.V[0] = vec3(1.0f, 1.0f, 1.0f);
            cube.V[1] = vec3(-1.0f, 1.0f, 1.0f);
            cube.V[2] = vec3(1.0f, -1.0f, 1.0f);
            cube.V[3] = vec3(-1.0f, -1.0f, 1.0f);
            cube.V[4] = vec3(1.0f, 1.0f, -1.0f);
            cube.V[5] = vec3(-1.0f, 1.0f, -1.0f);
            cube.V[6] = vec3(1.0f, -1.0f, -1.0f);
            cube.V[7] = vec3(-1.0f, -1.0f, -1.0f);

            const uchar F[] = { 0, 1, 3, 2, 4, 5, 7, 6, 0, 1, 5, 4, 2, 3, 7, 6, 0, 2, 6, 4, 1, 3, 7, 5 };
            memcpy(cube.F, F, sizeof(uchar) * 24);
            const uchar E[] = { 0, 1, 2, 3, 0, 2, 1, 3, 4, 5, 6, 7, 4, 6, 5, 7, 0, 4, 1, 5, 2, 6, 3, 7 };
            memcpy(cube.E, E, sizeof(uchar) * 24);

            cube.Prepare(kcd.vertScale * 0.5f);
        }

        // Set some random data
        {
            vec3 xi[kSDFMaxIterations] = {
                vec3(0.86418, 0.550822, 0.123257), vec3(0.39088, 0.450768, 0.883838), vec3(0.99579, 0.614493, 0.0399468), vec3(0.022739, 0.691733, 0.23853),
                vec3(0.0491209, 0.14045, 0.0436547), vec3(0.566151, 0.644241, 0.559143),  vec3(0.336939, 0.0539437, 0.244569), vec3(0.240348, 0.349121, 0.391118),
                vec3(0.0955733, 0.642528, 0.46847), vec3(0.188407, 0.360586, 0.659837)
            };
            memcpy(kcd.xi, xi, sizeof(vec3) * kSDFMaxIterations);
        }

        float rotateAlpha = mix(-1.0f, 1.0f, m_params.rotateA());
        rotateAlpha = 2.0 * powf(fabsf(rotateAlpha), 1.0f) * sign(rotateAlpha);
        float rotateBeta = mix(-1.0f, 1.0f, m_params.rotateB());
        rotateBeta = 2.0 * powf(fabsf(rotateBeta), 1.0f) * sign(rotateBeta);

        float scaleAlpha = mix(0.5, 1.5, m_params.scaleA());
        float scaleBeta = mix(-1.0, 1.0, m_params.scaleB());

        for (int i = 0; i < m_params.numIterations; i++)
        {
            float f = float(i + 1) / float(max(1, m_params.numIterations));
            vec3 theta = (2.0f * kcd.xi[i] - 1.0f) * kPi * 10.0f * (rotateAlpha + rotateBeta * f);
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
    __device__ __forceinline__ vec4 SDFPolyhedronFace(const vec3& p, const PolyType& poly, const uint& faceIdx)
    {
        const uchar* F = &poly.F[faceIdx * PolyType::kPolyOrder];
        const vec3* dV = &poly.dV[faceIdx * PolyType::kPolyOrder];
        const vec3* edgeNorm = &poly.edgeNorm[faceIdx * PolyType::kPolyOrder];
        vec3 grad;

        // Test against each of the polygon's edges
        for (int edge = 0; edge < PolyType::kPolyOrder; edge++)
        {
            const vec3& vi = poly.V[F[edge]];
            if (dot(edgeNorm[edge], p - vi) > 0.0f)
            {
                const vec3& dv = dV[edge];
                const float t = clamp((dot(p, dv) - dot(vi, dv)) / dot(dv, dv), 0.0f, 1.0f);
                grad = p - (vi + t * dv);
                const float gradMag = length(grad);
                return vec4(gradMag, grad / gradMag);
            }
        }

        // Test against the face itself
        const vec3& v0 = poly.V[F[0]];
        const vec3& N = poly.N[faceIdx];
        return (dot(N, p - v0) < 0.0f) ? vec4((dot(p, -N) - dot(v0, -N)), -N) : vec4((dot(p, N) - dot(v0, N)), N);
    }

    template<typename PolyType>
     __device__ __forceinline__ void SDFPolyhedronFace(vec4& F, const Device::KIFS::KernelConstantData& kcd, const float& iterationScale, const vec3& p, const PolyType& poly)
    {
         for (int i = 0; i < poly.kNumFaces; i++)
         {
             if ((kcd.faceMask & (1 << i)) != 0)
             {
                 vec4 FFace = SDFPolyhedronFace(p, poly, i);

                 FFace.x = (FFace.x * iterationScale) - kcd.crustThickness;
                 if (FFace.x < F.x) { F = FFace; }
             }
         }
    }

     template<typename PolyType>
     __device__ __forceinline__ void SDFPolyhedronCage(vec4& F, const Device::KIFS::KernelConstantData& kcd, const float& iterationScale, const vec3& p, const PolyType& poly)
     {
         for (int edge = 0; edge < poly.kNumEdges; edge++)
         {
             const vec3& vi = poly.V[poly.E[edge*2]];
             const vec3 dv = poly.V[poly.E[edge*2 + 1]] - vi;
             const float t = clamp((dot(p, dv) - dot(vi, dv)) / dot(dv, dv), 0.0f, 1.0f);
             const vec3 grad = p - (vi + t * dv);
             const float gradMag = length(grad);

             const float FEdge = (gradMag * iterationScale) - kcd.crustThickness;
             if (FEdge < F.x)
             {
                 F = vec4(FEdge, grad / gradMag);
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

            if (kcd.foldType == kKIFSFoldTetrahedron)
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
        switch(kcd.primitiveType)
        {
        case kKIFSPrimitiveTetrahedronSolid:
            SDFPolyhedronFace(F, kcd, iterationScale, p, kcd.tetrahedronData);
            break;
        case KIFSPrimitiveCubeSolid:
            SDFPolyhedronFace(F, kcd, iterationScale, p, kcd.cubeData);
            break;
        case KIFSPrimitiveSphere:
            F = SDFPrimitive::Sphere(p, 0.5f * kcd.vertScale);
            F.x = (F.x * iterationScale) - kcd.crustThickness;
            break;
        case KIFSPrimitiveTorus:
            F = SDFPrimitive::Torus(p, 0.35f * kcd.vertScale, 0.15f * kcd.vertScale);
            F.x = (F.x * iterationScale) - kcd.crustThickness;
            break;
        case kKIFSPrimitiveTetrahedronCage:
            SDFPolyhedronCage(F, kcd, iterationScale, p, kcd.tetrahedronData);
            break;
        case KIFSPrimitiveCubeCage:
            SDFPolyhedronCage(F, kcd, iterationScale, p, kcd.cubeData);
            break;
        default:
            F = SDFPrimitive::Box(p, 0.25f  * kcd.vertScale);
            F.x = (F.x * iterationScale) - kcd.crustThickness;
            break;
        }

        // Transform the normal from folded space into object space
        F.yzw = bi * F.yzw;
        F.yzw = normalize(basis[0] * F.y + basis[1] * F.z + basis[2] * F.w);
        pathId = code;

        return F;
    }

    __device__  bool Device::KIFS::Intersect(Ray& globalRay, HitCtx& hitCtx) const
    {
        if (globalRay.flags & kRayLightProbe && m_params.tracable.renderObject.flags() & kRenderObjectExcludeFromBake) { return false; }

        RayBasic localRay = RayToObjectSpace(globalRay.od, m_params.transform);

        float t = Intersector::RayBox(localRay, m_params.sdf.escapeThreshold);
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

            /*if(!(globalRay.flags & kRayLightProbe) && m_params.sdf.clipCameraRays && 
                (globalRay.depth == 1 || (globalRay.depth == 2 && globalRay.flags & kRayDirectSample)))*/
            if(m_params.sdf.flags() & kKIFSClipRays)
            {
                vec3 pWorld = globalRay.PointAt(t / localMag);
                vec4 FClip;
                switch (m_params.sdf.clipShape)
                {
                case kKIFSClipSphere:
                    FClip = SDFPrimitive::Sphere(pWorld, 0.5f); break;
                case kKIFSClipTorus:
                    FClip = SDFPrimitive::Torus(pWorld, 0.4f, 0.1f); break;
                default:
                    FClip = SDFPrimitive::Box(pWorld, 0.5f);
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
            else if (fabsf(F.x) < m_params.sdf.cutoffThreshold * localMag) { hitCtx.debug = vec3(0.0f, 0.0f, 1.0f) * float(i) / float(maxIterations); break; }

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
                    isSubsurface,
                    vec2(*reinterpret_cast<float*>(&code), 0.0f),  // Dump the bits of the code into the float. FIXME: Not type safe, so fix this
                    m_params.sdf.rayKickoff,
                    kNotALight
            );

        return true;
    }

    __host__ AssetHandle<Host::RenderObject> Host::KIFS::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kTracable) { return AssetHandle<Host::RenderObject>(); }

        return CreateAsset<Host::KIFS>(id, json);
    }

    __host__  Host::KIFS::KIFS(const std::string& id, const ::Json::Node& node) :
        Tracable(id),
        cu_deviceData(nullptr)
    {
        cu_deviceData = InstantiateOnDevice<Device::KIFS>(id);
        FromJson(node, ::Json::kSilent);
    }

    __host__ void Host::KIFS::OnDestroyAsset()
    {
        DestroyOnDevice(GetAssetID(), cu_deviceData);
    }

    __host__ void Host::KIFS::FromJson(const ::Json::Node& node, const uint flags)
    {
        Host::Tracable::FromJson(node, flags);
        
        m_params.FromJson(node, flags);
        RenderObject::SetUserFacingRenderObjectFlags(m_params.tracable.renderObject.flags());

        SynchroniseObjects(cu_deviceData, m_params);
    }
}