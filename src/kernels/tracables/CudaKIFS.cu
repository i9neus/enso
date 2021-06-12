#include "CudaKIFS.cuh"

namespace Cuda
{
#define kKIFSPolyhedronType 0
#define kKIFSFoldType       0

    constexpr uint kSDFMaxSteps = 100;
    constexpr uint kSDFMaxIterations = 10;
    constexpr float kSDFCutoffThreshold = 1e-4f;
    constexpr float kSDFEscapeThreshold = 1.0f;
    constexpr float kSDFRayIncrement = 0.7f;
    constexpr float kSDFFailThreshold = 1e-2f;

    __device__ Device::KIFS::KIFS(const BidirectionalTransform& transform) :
        Tracable(transform),
        m_tetrahedronData(4, 4 * 3, 3),
        m_cubeData(8, 6 * 4, 4)
    {
        {
            const vec3 V[4] = { vec3(1.0f, 1.0f, 1.0f) * 0.5f, vec3(-1.0f, -1.0f, 1.0f) * 0.5f, vec3(1.0f, -1.0f, -1.0f) * 0.5f, vec3(-1.0f, 1.0f, -1.0f) * 0.5f };
            const uint F[12] = { 0, 1, 2, 1, 2, 3, 2, 3, 0, 3, 0, 1 };
            memcpy((void*)m_tetrahedronData.V, V, sizeof(V));
            memcpy((void*)m_tetrahedronData.F, F, sizeof(F));
        }

        {
            const vec3 V[8] = { vec3(1.0f, 1.0f, 1.0f) * 0.5f, vec3(-1.0f, 1.0f, 1.0f) * 0.5f, vec3(1.0f, -1.0f, 1.0f) * 0.5f, vec3(-1.0f, -1.0f, 1.0f) * 0.5f,
                                          vec3(1.0f, 1.0f, -1.0f) * 0.5f, vec3(-1.0f, 1.0f, -1.0f) * 0.5f, vec3(1.0f, -1.0f, -1.0f) * 0.5f, vec3(-1.0f, -1.0f, -1.0f) * 0.5f };
            const uint F[6 * 4] = { 0, 1, 3, 2, 4, 5, 7, 6, 0, 1, 5, 4, 2, 3, 7, 6, 0, 2, 6, 4, 1, 3, 7, 5 };
            memcpy((void*)m_cubeData.V, V, sizeof(V));
            memcpy((void*)m_cubeData.F, F, sizeof(F));
        }
        
        switch (m_type)
        {
        case KIFSType::kTetrahedtron:
            m_polyData = &m_tetrahedronData; break;
        case KIFSType::kCube:
        default:
            m_polyData = &m_cubeData; break;
        };

        m_origin = m_polyData->F[0];
    }

    /*__device__ void Device::KIFS::FoldTetrahedron(const mat4& matrix, const int& i, vec3& p, mat3& bi, uint& code) const
    {
        if (p.x + p.y < 0.0)
        {
            p.xy = -p.xy;
            bi[0].xy = -bi[0].yx;
            bi[1].xy = -bi[1].yx;
            bi[2].xy = -bi[2].yx;
            code |= (1u << (3 * i));
        }
        if (p.x + p.z < 0.0)
        {
            p.xz = -p.zx;
            bi[0].xz = -bi[0].zx;
            bi[1].xz = -bi[1].zx;
            bi[2].xz = -bi[2].zx;
            code |= (1u << (3 * i + 1));
        }
        if (p.y + p.z < 0.0)
        {
            p.zy = -p.yz;
            bi[0].zy = -bi[0].yz;
            bi[1].zy = -bi[1].yz;
            bi[2].zy = -bi[2].yz;
            code |= (1u << (3 * i + 2));
        }
    }

    __device__ vec4 Device::KIFS::Field(vec3 p, const mat3& b, const RenderCtx& renderCtx, uint& code, uint& surfaceDepth) const
    {
        // TODO: Missing parameter
        //p.xz -= 5.0 * (gParamKIFSTranslate.xy - vec2(0.5));
        mat3 bi = b;
        mat3 bTrap = b;
        
        // TODO: Missing parameter
        //ivec2 iterations = ivec2(float(kSDFMaxIterations + 2) * vec2(gParamKIFSIterations1.y, gParamKIFSIterations2.y)) - ivec2(1.0);
        ivec2 iterations(1, 1);
        int maxIterations = max(iterations.x, iterations.y);

        vec2 kifsRotate = vec2(0.5) + vec2(cos(renderCtx.time), sin(renderCtx.time)) * 0.3;
        float rotateAlpha = mix(-1.0, 1.0, kifsRotate.x);
        rotateAlpha = 2.0 * powf(fabsf(rotateAlpha), 2.0) * sign(rotateAlpha);
        float rotateBeta = mix(-1.0, 1.0, kifsRotate.y);
        rotateBeta = 2.0 * powf(fabsf(rotateBeta), 2.0) * sign(rotateBeta);

        // TODO: Missing parameter
        //float scaleAlpha = mix(0.5, 1.5, gParamKIFSScale.x);
        //float scaleBeta = mix(-1.0, 1.0, gParamKIFSScale.y);
        float scaleAlpha = 1.0f;
        float scaleBeta = 1.0f;

        code = 0u;
        uint codeTrap = 0u;
        float iterationScale = 1.0;
        
        // TODO: Missing parameter
        //float thickness = powf(2.0, mix(-15.0, 0.0, gParamKIFSThickness.x));
        float thickness = 0.001f;

        // TODO: Missing parameter
        //bool hasOrbitTraps = gParamKIFSOrbitTraps.x > 0.0 && gParamKIFSOrbitTraps.y > 0.0;
        const bool hasOrbitTraps = false;

        // TODO: Missing parameter
        //int faceMask = int(float((2 << m_numFaces) - 1) * gParamKIFSDFFaces.y + 0.5);
        int faceMask = (2 << m_numFaces) - 1;

        // TODO: Missing parameter
        //float vertScale = pow(2.0, mix(-8.0, 8.0, gParamKIFSThickness.y));
        float vertScale = 1.0f;

        vec4 minTrap = vec4(kFltMax);
        vec4 trap = vec4(kFltMax);

        __constant__ static vec3 kXi[10] = { vec3(0.86418, 0.550822, 0.123257), vec3(0.39088, 0.450768, 0.883838),
                                                vec3(0.99579, 0.614493, 0.0399468), vec3(0.022739, 0.691733, 0.23853),
                                                vec3(0.0491209, 0.14045, 0.0436547), vec3(0.566151, 0.644241, 0.559143),
                                                vec3(0.336939, 0.0539437, 0.244569), vec3(0.240348, 0.349121, 0.391118),
                                                vec3(0.0955733, 0.642528, 0.46847), vec3(0.188407, 0.360586, 0.659837) };

        for (int i = 0; i <= kSDFMaxIterations; i++)
        {
            float f = float(i + 1) / float(max(1, maxIterations));
            vec3 theta = (2.0f * kXi[i] - 1.0f) * kPi * (rotateAlpha + rotateBeta * f);
            float iterScale = scaleAlpha * powf(2.0f, float(i) * scaleBeta);

            // TODO: Cache these matrices
            // compound(theta.x, theta.y, theta.z, vec3(iterScale), vec3(0.0, 0.0, 0.0));
            mat4 matrix = mat4::indentity();

            p = matrix * p;
            bi[0] = matrix * bi[0];
            bi[1] = matrix * bi[1];
            bi[2] = matrix * bi[2];
            iterationScale *= iterScale;

            bool trapped = false;
            if (i == iterations.x)
            {
                for (int j = 0; j < m_numFaces; j++)
                {
                    if ((faceMask & (1 << j)) != 0)
                    {
                        vec4 F = SDF::PolyhedronFace(p, m_V, m_numVertices, vertScale);

                        F.x = (F.x * iterationScale) - thickness;
                        if (F.x < trap.x) { trap = F; trapped = true; }
                    }
                }
            }

            if (i <= iterations.y)
            {
                vec4 F = SDF::Torus(p, vertScale, thickness);
                F.x = (F.x * iterationScale);
                if (F.x < trap.x) { trap = F; trapped = true; }
            }

            if (trapped)
            {
                bTrap = bi;
                codeTrap = code;
                surfaceDepth = uint(i);
            }

            if (i == maxIterations) { break; }

#if kKIFSFoldType == 0
            foldTetrahedron(matrix, i, p, bi, code);
#else
            foldCube(matrix, i, p, bi, code);
#endif         

            p = (p - kOne) * 2.0 + kOne;
            iterationScale *= 0.5;
        }

        trap.yzw *= bTrap;
        trap.yzw = normalize(b[0] * trap.y + b[1] * trap.z + b[2] * trap.w);

        code = codeTrap;

        return trap;
    }*/
    
    __device__  bool Device::KIFS::Intersect(Ray& ray, HitCtx& hitCtx) const
    {
        RayBasic localRay = RayToObjectSpace(ray.od, m_transform);

        float t = Intersector::RayUnitBox(localRay);
        if (t == kNoIntersect) { return false; }

        const float transformScale = 1.0f;
        const float localMag = length(localRay.d);
        localRay.d /= localMag;

        //const mat3 basis = CreateBasis(localRay.d);
        vec3 grad;
        vec3 p = localRay.PointAt(t);
        int i;
        vec4 F;
        bool isSubsurface = false;

        for (i = 0; i < kSDFMaxSteps; i++)
        {
            //F = field(p, basis, code, surfaceDepth);

            //if(!isOriginInBound)
            {
                //vec4 FBox = sdfBox(p, 0.5 / transformScale); 
                vec4 FBox = SDF::Torus(p, 0.25 / transformScale, 0.1 / transformScale);
                //vec4 FBox = SDF::Sphere(p, 0.5 / transformScale);
                //if (FBox.x > F.x) { F = FBox; }
                F = FBox;
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
        m_hostData.m_transform.MakeIdentity();

        cu_deviceData = InstantiateOnDevice<Device::KIFS>(m_hostData.m_transform);
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