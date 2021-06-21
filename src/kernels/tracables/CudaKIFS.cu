#include "CudaKIFS.cuh"
#include "CudaSDF.cuh"

namespace Cuda
{
    #define kKIFSPolyhedronType 0
    #define kKIFSFoldType       0

    constexpr uint kSDFMaxSteps = 100;
    constexpr float kSDFCutoffThreshold = 1e-4f;
    constexpr float kSDFEscapeThreshold = 1.0f;
    constexpr float kSDFRayIncrement = 0.7f;
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

    __device__ void Device::KIFS::Initialise()
    {
        m_kernelConstantData.numIterations = m_params.iterations;
        m_kernelConstantData.vertScale = powf(2.0, mix(-8.0f, 8.0f, m_params.vertScale));
        m_kernelConstantData.isosurfaceThickness = powf(2.0f, mix(-15.0f, 0.0f, m_params.thickness));
        m_kernelConstantData.faceMask = m_params.faceMask;

        float rotateAlpha = mix(-1.0f, 1.0f, m_params.rotate.x);
        rotateAlpha = 2.0 * powf(fabsf(rotateAlpha), 2.0f) * sign(rotateAlpha);
        float rotateBeta = mix(-1.0f, 1.0f, m_params.rotate.y);
        rotateBeta = 2.0 * powf(fabsf(rotateBeta), 2.0f) * sign(rotateBeta);

        float scaleAlpha = mix(0.5, 1.5, m_params.scale.x);
        float scaleBeta = mix(-1.0, 1.0, m_params.scale.y);

        for (int i = 0; i < kSDFMaxIterations; i++)
        {
            float f = float(i + 1) / float(max(1, m_maxIterations));
            vec3 theta = (2.0f * kXi[i] - 1.0f) * kPi * (rotateAlpha + rotateBeta * f);
            float iterScale = scaleAlpha * powf(2.0f, float(i) * scaleBeta);

            m_kernelConstantData.iteration[i].matrix = CreateCompoundTransform(vec3(theta), vec3(iterScale)).fwd;
            m_kernelConstantData.iteration[i].scale = iterScale;
        }
    }

    __device__ Device::KIFS::KIFS() :
        Tracable(CreateCompoundTransform(vec3(0.0f), vec3(0.0f, 0.5f, 0.0f)))
    {        
        m_type = KIFSType::kTetrahedtron;     

        Initialise();
    }

    __device__ Device::KIFS::KernelConstantData& GetKernelConstantData()
    {
        __device__ __shared__ Device::KIFS::KernelConstantData kcd;
        return kcd;
    }

    __device__ void Device::KIFS::Precache()
    {
        if (kThreadIdx == 0)
        {
            memcpy(&GetKernelConstantData(), &m_kernelConstantData, sizeof(KernelConstantData));
        }
        __syncthreads();
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
        const vec3& N = poly.N[faceIdx];
        const uchar* F = &poly.F[faceIdx * PolyType::kPolyOrder];
        vec3 grad;

        vec4 A;
        A.x = kFltMax;
        for (int i = 0; i < PolyType::kPolyOrder; i++)
        {
            auto v0 = V[F[(i + 1) % poly.kPolyOrder]];
            auto v1 = V[F[i]];
            vec4 Ai = SDF::Capsule(p, v0, v1, 0.01f);
            if (Ai.x < A.x) { A = Ai; }
        }
        return A;
        // Test against each of the polygon's edges
        /*for (int i = 0; i < PolyType::kPolyOrder; i++)
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
        return (dot(N, p - v0) < 0.0f) ? vec4((dot(p, -N) - dot(v0, -N)), -N) : vec4((dot(p, N) - dot(v0, N)), N);*/
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

        /*for (i = 0; i < scratch.global.numIterations; i++)
        {
            const mat3& matrix = scratch.iteration[i].matrix;
            const float& iterScale = scratch.iteration[i].scale;

            p = matrix * p;
            bi[0] = matrix * bi[0];
            bi[1] = matrix * bi[1];
            bi[2] = matrix * bi[2];
            iterationScale *= iterScale;           

            if (i == scratch.global.numIterations - 1) { break; }

#if kKIFSFoldType == 0
            FoldTetrahedron(matrix, i, p, bi, code);
#else
            foldCube(matrix, i, p, bi, code);
#endif         

            p = (p - kOne) * 2.0f + kOne;
            iterationScale *= 0.5f;
        }
        
        // Transform the normal from folded space into object space
        F.yzw = bi * F.yzw;
        F.yzw = normalize(basis[0] * F.y + basis[1] * F.z + basis[2] * F.w);
        pathId = code;
        */

        /*SimplePolyhedron<4, 4, 3> tet = {
                { vec3(1.0f, 1.0f, 1.0f) * 0.5f, vec3(-1.0f, -1.0f, 1.0f) * 0.5f, vec3(1.0f, -1.0f, -1.0f) * 0.5f, vec3(-1.0f, 1.0f, -1.0f) * 0.5f },
                { 0, 1, 2, 1, 2, 3, 2, 3, 0, 3, 0, 1 },
                { vec3(0.0f), vec3(0.0f), vec3(0.0f), vec3(0.0f) }
        };
        tet.Prepare();*/

        // Test this position against each polyhedron face
        F.x = kFltMax;
        for (i = 0; i < kCubeData.kNumFaces; i++)
        {
            //if ((kcd.faceMask & (1 << i)) != 0)
            {
                vec4 FFace = PolyhedronFace(p, kCubeData, i, kcd.vertScale);                

                FFace.x = (FFace.x * iterationScale) - m_isosurfaceThickness;
                if (FFace.x < F.x) { F = FFace; }
            }
        }        

        //F= SDF::Torus(position, 0.3, 0.1);
      
        return F;
    }
    
    __device__  bool Device::KIFS::Intersect(Ray& ray, HitCtx& hitCtx) const
    {
        RayBasic localRay = RayToObjectSpace(ray.od, m_transform);

        //float t = Intersector::RayUnitBox(localRay);
        //if (t == kNoIntersect) { return false; }
        float t = 0.0f;

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
                //vec4 FBox = SDF::Box(p, 0.5f); 
                //vec4 FBox = SDF::Torus(p, 0.25 / transformScale, 0.1 / transformScale);
                //vec4 FBox = SDF::Sphere(p, 0.5 / transformScale);
                //if (FBox.x > F.x) { F = FBox; } 
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

    __host__  Host::KIFS::KIFS()
        : cu_deviceData(nullptr)
    {
        cu_deviceData = InstantiateOnDevice<Device::KIFS>();

        {
            SimplePolyhedron<4, 4, 3> tet = {
                { vec3(1.0f, 1.0f, 1.0f) * 0.5f, vec3(-1.0f, -1.0f, 1.0f) * 0.5f, vec3(1.0f, -1.0f, -1.0f) * 0.5f, vec3(-1.0f, 1.0f, -1.0f) * 0.5f },
                { 0, 1, 2, 1, 2, 3, 2, 3, 0, 3, 0, 1 },
                { vec3(0.0f), vec3(0.0f), vec3(0.0f), vec3(0.0f) }
            };
            tet.Prepare();
            IsOk(cudaMemcpyToSymbol(kTetrahedronData, &tet, sizeof(SimplePolyhedron<4, 4, 3>)));
        }

        {
            SimplePolyhedron<8, 6, 4> tet = {
               { vec3(1.0f, 1.0f, 1.0f) * 0.5f, vec3(-1.0f, 1.0f, 1.0f) * 0.5f, vec3(1.0f, -1.0f, 1.0f) * 0.5f, vec3(-1.0f, -1.0f, 1.0f) * 0.5f,
                 vec3(1.0f, 1.0f, -1.0f) * 0.5f, vec3(-1.0f, 1.0f, -1.0f) * 0.5f, vec3(1.0f, -1.0f, -1.0f) * 0.5f, vec3(-1.0f, -1.0f, -1.0f) * 0.5f },
               { 0, 1, 3, 2, 4, 5, 7, 6, 0, 1, 5, 4, 2, 3, 7, 6, 0, 2, 6, 4, 1, 3, 7, 5 },
               {vec3(0.0f), vec3(0.0f), vec3(0.0f), vec3(0.0f), vec3(0.0f), vec3(0.0f)}
            };
            tet.Prepare();
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

    __host__ void Host::KIFS::OnJson(const Json::Node& jsonNode)
    {
        Device::KIFS::Params params;

        //jsonNode.GetVector("albedo", params.albedo, true);

        //Log::Debug("albedo: %s", params.albedo.format());

        SyncParameters(cu_deviceData, params);
    }
}