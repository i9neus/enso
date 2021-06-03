#include "CudaKIFS.cuh"
#include "CudaSDF.cuh"

namespace Cuda
{
#define kKIFSPolyhedronType 0
#define kKIFSFoldType       0

    constexpr uint kSDFMaxSteps = 100;
    constexpr uint kSDFMaxIterations = 10;
    constexpr float kSDFCutoffThreshold = 1e-4f;
    constexpr float kSDFEscapeThreshold = 10.0f;
    constexpr float kSDFRayIncrement = 0.7f;
    constexpr float kSDFFailThreshold = 1e-2f;

    __device__ void GetTetrahedronConstData(__constant__ vec3** V, __constant__ uint** F, uint* numVertices, uint* numFaces, uint* polyOrder)
    {
        __constant__ static vec3 kV[6] = { vec3(1.0f, 1.0f, 1.0f) * 0.5f, vec3(-1.0f, -1.0f, 1.0f) * 0.5f, vec3(1.0f, -1.0f, -1.0f) * 0.5f, vec3(-1.0f, 1.0f, -1.0f) * 0.5f };
        __constant__ static uint kF[4 * 4] = { 0, 1, 2, 0, 1, 2, 3, 0, 2, 3, 0, 0, 3, 0, 1, 0 };
        
        *V = kV;
        *F = kF;
        *numVertices = 4u;
        *numFaces = 4u;
        *polyOrder = 3u;
    }

    __device__ void GetCubeConstData(__constant__ vec3** V, __constant__ uint** F, uint* numVertices, uint* numFaces, uint* polyOrder)
    {
        __constant__ static vec3 kV[8] = { vec3(1.0f, 1.0f, 1.0f) * 0.5f, vec3(-1.0f, 1.0f, 1.0f) * 0.5f, vec3(1.0f, -1.0f, 1.0f) * 0.5f, vec3(-1.0f, -1.0f, 1.0f) * 0.5f,
                                          vec3(1.0f, 1.0f, -1.0f) * 0.5f, vec3(-1.0f, 1.0f, -1.0f) * 0.5f, vec3(1.0f, -1.0f, -1.0f) * 0.5f, vec3(-1.0f, -1.0f, -1.0f) * 0.5f };
        __constant__ static uint kF[6 * 4] = { 0, 1, 3, 2, 4, 5, 7, 6, 0, 1, 5, 4, 2, 3, 7, 6, 0, 2, 6, 4, 1, 3, 7, 5 };

        *V = kV;
        *F = kF;
        *numVertices = 8u;
        *numFaces = 6u;
        *polyOrder = 4u;
    }

    __host__ Device::KIFS::KIFS(const mat4Pair& transform, const KIFSType& type) :
        Tracable(transform)
    {
        m_type = type;

        switch (m_type)
        {
        case KIFSType::kTetrahedtron:
            GetTetrahedronConstData(&m_V, &m_F, &m_numVertices, &m_numFaces, &m_polyOrder); break;
        case KIFSType::kCube:
            GetCubeConstData(&m_V, &m_F, &m_numVertices, &m_numFaces, &m_polyOrder); break;
        };

        m_origin = m_V[0];
    }

    __device__ void Device::KIFS::FoldTetrahedron(const mat4& matrix, const int& i, vec3& p, mat3& bi, uint& code) const
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
        /*float instance = floor(gParamKIFSInstance.y * 3.0 + 0.5);
        if (instance > 0.0)
        {
            instance = 1.0 + 2.0 * (instance - 1.0);
            vec3 q;
            if (p.x >= 0.0) { q.x = (p.x < instance) ? fract(p.x) : (p.x - instance); }
            else
            {
                q.x = (p.x > -instance) ? fract(abs(p.x)) : (instance - p.x);
                bi[0].x = -bi[0].x;
                bi[1].x = -bi[1].x;
                bi[2].x = -bi[2].x;
            }
            if (abs(int(p.x)) % 2 == 1)
            {
                q.x = 1.0 - q.x;
                bi[0].x = -bi[0].x;
                bi[1].x = -bi[1].x;
                bi[2].x = -bi[2].x;
            }
            if (p.z >= 0.0) { q.z = (p.z < instance) ? fract(p.z) : (p.z - instance); }
            else
            {
                q.z = (p.z > -instance) ? fract(abs(p.z)) : (instance - p.z);
                bi[0].z = -bi[0].z;
                bi[1].z = -bi[1].z;
                bi[2].z = -bi[2].z;
            }
            if (abs(int(p.z)) % 2 == 1)
            {
                q.z = 1.0 - q.z;
                bi[0].z = -bi[0].z;
                bi[1].z = -bi[1].z;
                bi[2].z = -bi[2].z;
            }
            p = vec3(q[0] - 0.5, p[1], q[2] - 0.5);
        }*/
        
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
    }
    
    __device__  bool Device::KIFS::Intersect(Ray& ray, HitCtx& hitCtx) const
    {
        Ray::Basic localRay = ray.od.ToObjectSpace(m_matrix);
        
        const float transformScale = 1.0f;
        const float localMag = length(localRay.d);
        localRay.d /= localMag;

        float s = sign(localRay.d.z);
        float a = -1.0 / (s + localRay.d.z);
        float b = localRay.d.x * localRay.d.y * a;
        mat3 basis;
        basis[0] = localRay.d;
        basis[1] = vec3(1.0f + s * localRay.d.x * localRay.d.x * a, s * b, -s * localRay.d.x);
        basis[2] = vec3(b, s + localRay.d.y * localRay.d.y * a, -localRay.d.y);

        vec3 grad;
        vec3 p = localRay.o;
        int i;
        float t;
        vec4 F;

        bool isBounded = false;
        bool isSubsurface = false;
        bool isOriginInBound = cwiseMax(abs(localRay.o)) < 0.5 / transformScale;

        for (i = 0; i < kSDFMaxSteps; i++)
        {
            F = field(p, basis, code, surfaceDepth);

            //if(!isOriginInBound)
            {
                //vec4 FBox = sdfBox(p, 0.5 / transformScale); 
                //vec4 FBox = sdfTorus(p, 0.5 / transformScale, 0.2 / transformScale);
                vec4 FBox = SDF::Sphere(p, 0.5 / transformScale);
                if (FBox.x > F.x) { F = FBox; }
            }

            //F = sdfTorus(p, 0.5, 0.2);
            //F = sdfSphere(p, 0.5);

            // On the first iteration, simply determine whether we're inside the isosurface or not
            if (i == 0) { isSubsurface = F.x < 0.0; }
            // Otherwise, check to see if we're at the surface
            else if (F.x > 0.0 && F.x < kSDFCutoffThreshold) { break; }

            if (!isBounded) { if (F.x < kSDFEscapeThreshold) { isBounded = true; } }
            else if (F.x > kSDFEscapeThreshold) { return nullRay(); }

            t += isSubsurface ? -F.x : F.x;
            p = localRay.o + t * localRay.d;
        }

        t /= localMag;
        if (F.x > kSDFFailThreshold || t > ray.tNear) { return nullRay(); }

        // If we've hit the surface and it's the closest intersection, calculate the normal and UV coordinates
        // A more efficient way would be to defer this process to avoid unncessarily computing normals for occuluded surfaces.
        const vec3 hitLocal = localRay.o + localRay.d * t;
        const vec3 hitGlobal = m_invMatrix * hitLocal;
        vec3 nGlobal = normalize((m_invMatrix * (n + hitLocal)) - hitGlobal);
        if (dot(n, localRay.d) > 0.0) { nGlobal = -nGlobal; }

        ray.tNear = t;
        hitCtx.Set(nGlobal, false, uv, 1e-5f);

        return true;
    }


    __host__  Host::KIFS::KIFS()
        : cu_deviceData(nullptr)
    {
        m_hostData.m_matrix = mat4::indentity();
        m_hostData.m_invMatrix = mat4::indentity();

        InstantiateOnDevice(&cu_deviceData, m_hostData.m_matrix, m_hostData.m_invMatrix);
    }

    __host__ void Host::KIFS::OnDestroyAsset()
    {
        DestroyOnDevice(&cu_deviceData);
    }

}