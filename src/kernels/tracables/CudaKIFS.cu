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

    __device__ Device::KIFS::KIFS() :
        Tracable(CreateCompoundTransform(vec3(0.0f), vec3(0.0f, 0.5f, 0.0f))),
        m_kXi{ vec3(0.86418, 0.550822, 0.123257), vec3(0.39088, 0.450768, 0.883838), vec3(0.99579, 0.614493, 0.0399468), vec3(0.022739, 0.691733, 0.23853),
               vec3(0.0491209, 0.14045, 0.0436547), vec3(0.566151, 0.644241, 0.559143),  vec3(0.336939, 0.0539437, 0.244569), vec3(0.240348, 0.349121, 0.391118),
               vec3(0.0955733, 0.642528, 0.46847), vec3(0.188407, 0.360586, 0.659837) },
        m_tetrahedronData{ { vec3(1.0f, 1.0f, 1.0f) * 0.5f, vec3(-1.0f, -1.0f, 1.0f) * 0.5f, vec3(1.0f, -1.0f, -1.0f) * 0.5f, vec3(-1.0f, 1.0f, -1.0f) * 0.5f },
        { 0, 1, 2, 1, 2, 3, 2, 3, 0, 3, 0, 1 } },
        m_cubeData{ { vec3(1.0f, 1.0f, 1.0f) * 0.5f, vec3(-1.0f, 1.0f, 1.0f) * 0.5f, vec3(1.0f, -1.0f, 1.0f) * 0.5f, vec3(-1.0f, -1.0f, 1.0f) * 0.5f,
                                          vec3(1.0f, 1.0f, -1.0f) * 0.5f, vec3(-1.0f, 1.0f, -1.0f) * 0.5f, vec3(1.0f, -1.0f, -1.0f) * 0.5f, vec3(-1.0f, -1.0f, -1.0f) * 0.5f },  
            { 0, 1, 3, 2, 4, 5, 7, 6, 0, 1, 5, 4, 2, 3, 7, 6, 0, 2, 6, 4, 1, 3, 7, 5 } }

    {        
        m_type = KIFSType::kTetrahedtron;
        switch (m_type)
        {
        case KIFSType::kTetrahedtron:
            m_origin = m_tetrahedronData.F[0];  break;
        case KIFSType::kCube:
        default:
            m_origin = m_cubeData.F[0]; break;
        };

        Initialise();
    }

    __device__ void Device::KIFS::Initialise()
    {
        m_iterations = m_params.iterations;
        m_maxIterations = cwiseMax(m_params.iterations);
        m_vertScale = powf(2.0, mix(-8.0f, 8.0f, m_params.vertScale));
        m_isosurfaceThickness = powf(2.0f, mix(-15.0f, 0.0f, m_params.thickness));
        m_faceMask = m_params.faceMask;
        
        float rotateAlpha = mix(-1.0f, 1.0f, m_params.rotate.x);
        rotateAlpha = 2.0 * powf(fabsf(rotateAlpha), 2.0f) * sign(rotateAlpha);
        float rotateBeta = mix(-1.0f, 1.0f, m_params.rotate.y);
        rotateBeta = 2.0 * powf(fabsf(rotateBeta), 2.0f) * sign(rotateBeta);

        float scaleAlpha = mix(0.5, 1.5, m_params.scale.x);
        float scaleBeta = mix(-1.0, 1.0, m_params.scale.y);

        for (int i = 0; i <= kSDFMaxIterations; i++)
        {
            float f = float(i + 1) / float(max(1, m_maxIterations));
            vec3 theta = (2.0f * m_kXi[i] - 1.0f) * kPi * (rotateAlpha + rotateBeta * f);
            float iterScale = scaleAlpha * powf(2.0f, float(i) * scaleBeta);

            m_matrices[i] = CreateCompoundTransform(vec3(theta), vec3(iterScale)).fwd;
            m_iterScales[i] = iterScale;
        }
    }

    __device__ void Device::KIFS::FoldTetrahedron(const mat3& matrix, const int& i, vec3& p, mat3& bi, uint& code) const
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

    __device__ vec4 Device::KIFS::Field(vec3 p, const mat3& b, uint& code, uint& surfaceDepth) const
    {
        mat3 bi = b;
        mat3 bTrap = b;
        code = 0u;
        uint codeTrap = 0u;
        float iterationScale = 1.0f;             

        vec4 minTrap = vec4(kFltMax);
        vec4 trap = vec4(kFltMax);

        for (int i = 0; i <= kSDFMaxIterations; i++)
        {
            const mat3& matrix = m_matrices[i];
            const float& iterScale = m_iterScales[i];

            p = matrix * p;
            bi[0] = matrix * bi[0];
            bi[1] = matrix * bi[1];
            bi[2] = matrix * bi[2];
            iterationScale *= iterScale;

            bool trapped = false;
            if (i == m_iterations.x)
            {
                for (int j = 0; j < m_numFaces; j++)
                {
                    if ((m_faceMask & (1 << j)) != 0)
                    {
                        vec4 F = SDF::PolyhedronFace(p, m_tetrahedronData, m_numVertices, m_vertScale);

                        F.x = (F.x * iterationScale) - m_isosurfaceThickness;
                        if (F.x < trap.x) { trap = F; trapped = true; }
                    }
                }
            }

            if (i <= m_iterations.y)
            {
                vec4 F = SDF::Torus(p, m_vertScale, m_isosurfaceThickness);
                F.x = (F.x * iterationScale);
                if (F.x < trap.x) { trap = F; trapped = true; }
            }

            if (trapped)
            {
                bTrap = bi;
                codeTrap = code;
                surfaceDepth = uint(i);
            }

            if (i == m_maxIterations) { break; }

#if kKIFSFoldType == 0
            FoldTetrahedron(matrix, i, p, bi, code);
#else
            foldCube(matrix, i, p, bi, code);
#endif         

            p = (p - kOne) * 2.0f + kOne;
            iterationScale *= 0.5f;
        }

        trap.yzw = bTrap * trap.yzw;
        trap.yzw = normalize(b[0] * trap.y + b[1] * trap.z + b[2] * trap.w);

        code = codeTrap;

        return trap;
    }
    
    __device__  bool Device::KIFS::Intersect(Ray& ray, HitCtx& hitCtx) const
    {
        RayBasic localRay = RayToObjectSpace(ray.od, m_transform);

        float t = Intersector::RayUnitBox(localRay);
        if (t == kNoIntersect) { return false; }

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
                vec4 FBox = SDF::Box(p, 0.45 / transformScale); 
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

    __host__  Host::KIFS::KIFS()
        : cu_deviceData(nullptr)
    {
        cu_deviceData = InstantiateOnDevice<Device::KIFS>();
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