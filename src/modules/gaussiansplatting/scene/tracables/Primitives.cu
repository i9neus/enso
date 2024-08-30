#include "Primitives.cuh"

#include "core/3d/primitives/GenericIntersector.cuh"
#include "core/3d/Ctx.cuh"
#include "core/3d/Transform.cuh"
#include "core/math/Mappings.cuh"

#include "../pointclouds/GaussianPointCloud.cuh"
#include "core/math/samplers/MersenneTwister.cuh"

namespace Enso
{
    template<>
    __device__ bool Device::Primitive<PlaneParams>::IntersectRay(Ray& ray, HitCtx& hit) const
    {
        const RayBasic localRay = Tracable::m_params.transform.RayToObjectSpace(ray.od);

        if (m_params.hideBackfacing && localRay.o.z < 0.f) { return 0.; }

        const float t = Intersector::RayPlane(localRay);
        if (t <= 0.0 || t >= ray.tNear)
        {
            return false;
        }
        else
        {
            const vec2 uv = (localRay.o.xy + localRay.d.xy * t) + 0.5f;
            if (m_params.isBounded && (cwiseMin(uv) < 0.0 || cwiseMax(uv) > 1.0))
            {
                return false;
            }
            else
            {
                ray.tNear = t;
                ray.SetFlag(kRayBackfacing, localRay.o.z < 0.0f);
                hit.n = Tracable::m_params.transform.NormalToWorldSpace(vec3(0.0, 0.0, 1.0));
                hit.uv = uv;
                return true;
            }
        }
    }

    template<>
    __device__ bool Device::Primitive<UnitSphereParams>::IntersectRay(Ray& ray, HitCtx& hit) const
    {
        const RayBasic localRay = Tracable::m_params.transform.RayToObjectSpace(ray.od);

        vec2 t;
        if (!Intersector::RayUnitSphere(localRay, t))
        {
            return false;
        }
        else
        {
            if (t.y < t.x) { swap(t.x, t.y); }

            vec3 n;
            float tNear = ray.tNear;
            if (t.x > 0.0 && t.x < tNear)
            {
                n = localRay.PointAt(t.x);
                tNear = t.x;
            }
            else if (t.y > 0.0 && t.y < tNear)
            {
                n = localRay.PointAt(t.y);
                tNear = t.y;
            }
            else { return false; }

            ray.tNear = tNear;
            ray.SetFlag(kRayBackfacing, dot(localRay.o, localRay.o) < 1.0);
            hit.n = Tracable::m_params.transform.NormalToWorldSpace(n);

            return true;
        }
    }

    template<>
    __device__ bool Device::Primitive<CylinderParams>::IntersectRay(Ray& ray, HitCtx& hit) const
    {
        const RayBasic localRay = Tracable::m_params.transform.RayToObjectSpace(ray.od);

        // A ray intersects a sphere in at most two places which means we can find t by solving a quadratic
        float a = dot(localRay.d.xy, localRay.d.xy);
        float b = 2.0 * dot(localRay.d.xy, localRay.o.xy);
        float c = dot(localRay.o.xy, localRay.o.xy) - 1.0;

        // Intersect the unbounded cylinder
        vec2 tNearCyl, tFarCyl;
        float b2ac4 = b * b - 4.0f * a * c;
        if (b2ac4 < 0.0) { return false; }
        b2ac4 = sqrt(b2ac4);
        tNearCyl = (-b + b2ac4) / (2.0f * a);
        tFarCyl = (-b - b2ac4) / (2.0f * a);
        sort(tNearCyl.x, tFarCyl.x);

        // Intersect the caps
        tNearCyl.y = (-m_params.height * 0.5 - localRay.o.z) / localRay.d.z;
        tFarCyl.y = (m_params.height * 0.5 - localRay.o.z) / localRay.d.z;
        sort(tNearCyl.y, tFarCyl.y);

        float tNearMax = fmaxf(tNearCyl.x, tNearCyl.y);
        float tFarMin = fminf(tFarCyl.x, tFarCyl.y);
        if (tNearMax > tFarMin) { return false; }  // Ray didn't hit 

        float tNear;
        if (tNearMax > 0.0 && tNearMax < ray.tNear) { tNear = tNearMax; }
        else if (tFarMin > 0.0 && tFarMin < ray.tNear) { tNear = tFarMin; }
        else { return false; } // Box is behind the ray

        const vec3 i = localRay.PointAt(tNear);
        const bool hitCap = tNearCyl.x < tNearCyl.y;

        ray.tNear = tNear;
        hit.n = Tracable::m_params.transform.NormalToWorldSpace(hitCap ? vec3(0.0, 0.0, sign(i.z)) : vec3(i.xy, 0.));
        // TODO: Computing this here is expensive. Ideally we should only do it after all the primitives have been intersected.
        hit.uv = hitCap ? vec2(i.xy) : vec2((atan2f(i.y, i.x) + kPi) / kTwoPi, (i.z + m_params.height * 0.5) / m_params.height);
        ray.SetFlag(kRayBackfacing, dot(localRay.o, localRay.o) < 1.0);

        return true;
    }

    template<>
    __device__ bool Device::Primitive<BoxParams>::IntersectRay(Ray& ray, HitCtx& hit) const
    {
        const RayBasic localRay = Tracable::m_params.transform.RayToObjectSpace(ray.od);

        vec3 tNearPlane, tFarPlane;
        for (int dim = 0; dim < 3; dim++)
        {
            if (fabsf(localRay.d[dim]) > 1e-10)
            {
                float t0 = (m_params.dims[dim] * 0.5f - localRay.o[dim]) / localRay.d[dim];
                float t1 = (-m_params.dims[dim] * 0.5f - localRay.o[dim]) / localRay.d[dim];
                if (t0 < t1) { tNearPlane[dim] = t0;  tFarPlane[dim] = t1; }
                else { tNearPlane[dim] = t1;  tFarPlane[dim] = t0; }
            }
        }

        float tNearMax = cwiseMax(tNearPlane);
        float tFarMin = cwiseMin(tFarPlane);
        if (tNearMax > tFarMin) { return false; }  // Ray didn't hit the box

        float tNear;
        if (tNearMax > 0.0) { tNear = tNearMax; }
        else if (tFarMin > 0.0) { tNear = tFarMin; }
        else { return false; } // Box is behind the ray

        if (tNear > ray.tNear) { return false; }

        vec3 hitLocal = localRay.o + localRay.d * tNear;
        int normPlane = (fabsf(hitLocal.x / m_params.dims.x) > fabsf(hitLocal.y / m_params.dims.y)) ?
            ((fabsf(hitLocal.x / m_params.dims.x) > fabsf(hitLocal.z / m_params.dims.z)) ? 0 : 2) :
            ((fabsf(hitLocal.y / m_params.dims.y) > fabsf(hitLocal.z / m_params.dims.z)) ? 1 : 2);

        vec3 n = kZero;
        n[normPlane] = sign(hitLocal[normPlane]);

        ray.tNear = fmaxf(0.0f, tNear);
        hit.n = Tracable::m_params.transform.NormalToWorldSpace(n);
        ray.SetFlag(kRayBackfacing, cwiseMax(abs(localRay.o)) < m_params.dims[normPlane]);

        return true;
    }

    template<typename ParamsType>
    __host__ Host::Primitive<ParamsType>::Primitive(const InitCtx& initCtx, const BidirectionalTransform& transform, const int materialIdx, const ParamsType& params) :
        Tracable(initCtx, transform, materialIdx),
        cu_deviceInstance(AssetAllocator::InstantiateOnDevice<DeviceType>(*this))
    {
        Tracable::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::Tracable>(cu_deviceInstance));
        
        m_params = params;
        Synchronise(kSyncParams);
    }

    template<typename ParamsType>
    __host__ Host::Primitive<ParamsType>::~Primitive()
    {
        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);
    }

    template<typename ParamsType>
    __host__ GaussianPoint Host::Primitive<ParamsType>::GenerateRandomGaussianPoint(const vec3& p, float gaussSigma, MersenneTwister& rng) const
    {
        GaussianPoint pt;
        pt.p = p;
        pt.rot = normalize(mix(vec4(-1.f), vec4(1.f), rng.Rand4()));
        pt.sca = gaussSigma * mix(vec3(0.1f), vec3(1.0f), rng.Rand3());
        pt.rgba = vec4(rng.Rand3(), 1.0f);
        return pt;
    }

    template<>
    __host__ std::vector<GaussianPoint> Host::Primitive<PlaneParams>::GenerateGaussianPointCloud(const int numPoints, const float areaGain, MersenneTwister& rng)
    {
        if (!m_params.isBounded)
        {
            Log::Error("Warning: cannot create points on an unbounded plane!");
            return std::vector<GaussianPoint>();
        }

        const float gaussSigma = areaGain * std::sqrt(CalculateSurfaceArea() / numPoints);

        std::vector<GaussianPoint> points(numPoints);
        const vec3 n = Tracable::m_params.transform.NormalToWorldSpace(vec3(0.0f, 0.0f, 1.0f));

        for (auto& pt : points)
        {
            const vec3 p = vec3(mix(vec2(-.5f, -.5f), vec2(.5f, .5f), rng.Rand2()), 0.f);
            pt = GenerateRandomGaussianPoint(Tracable::m_params.transform.PointToWorldSpace(p), gaussSigma, rng);
        }

        return points;
    }

    template<>
    __host__ std::vector<GaussianPoint> Host::Primitive<UnitSphereParams>::GenerateGaussianPointCloud(const int numPoints, const float areaGain, MersenneTwister& rng)
    {
        const float gaussSigma = areaGain * std::sqrt(CalculateSurfaceArea() / numPoints);

        std::vector<GaussianPoint> points(numPoints);
        for (auto& pt : points)
        {
            const vec3 p = SampleUnitSphere(rng.Rand2());
            pt = GenerateRandomGaussianPoint(Tracable::m_params.transform.PointToWorldSpace(p), gaussSigma, rng);
        }

        return points;
    }

    template<>
    __host__ std::vector<GaussianPoint> Host::Primitive<CylinderParams>::GenerateGaussianPointCloud(const int numPoints, const float areaGain, MersenneTwister& rng)
    {
        const float gaussSigma = areaGain * std::sqrt(CalculateSurfaceArea() / numPoints);

        std::vector<GaussianPoint> points(numPoints);

        // Build a small CMF to sample the caps or the side
        float cmf[3] = { kPi, kPi, 2 * kPi * m_params.height };
        const float sum = cmf[0] + cmf[1] + cmf[2];
        cmf[0] /= sum;
        cmf[1] /= sum;

        for (auto& pt : points)
        {
            // Sample the PMF to determine which face we're creating a splat point on
            vec3 xi = rng.Rand3();
            const int faceIdx = (xi.z < cmf[0]) ? 0 : ((xi.z < cmf[1]) ? 1 : 2);
            vec3 p;

            if (faceIdx < 2)
            {
                // Sample the caps
                vec2 uv = SampleUnitDisc(xi.xy);
                p = vec3(uv, m_params.height * (faceIdx * 2 - 1) * 0.5f);
            }
            else
            {
                // Sample the side
                const float theta = kTwoPi * xi.x;
                p = vec3(vec2(cos(theta), sin(theta)), mix(-1., 1., xi.y) * m_params.height * 0.5f);
            }

            // Create the splat
            pt = GenerateRandomGaussianPoint(Tracable::m_params.transform.PointToWorldSpace(p), gaussSigma, rng);
        }

        return points;
    }

    template<>
    __host__ std::vector<GaussianPoint> Host::Primitive<BoxParams>::GenerateGaussianPointCloud(const int numPoints, const float areaGain, MersenneTwister& rng)
    {
        const float gaussSigma = areaGain * std::sqrt(CalculateSurfaceArea() / numPoints);

        std::vector<GaussianPoint> points(numPoints);

        // Build a small CMF based on the face area of each dimension
        float cmf[3] = { m_params.dims[1] * m_params.dims[2], m_params.dims[2] * m_params.dims[0], m_params.dims[0] * m_params.dims[1] };
        const float sum = cmf[0] + cmf[1] + cmf[2];
        cmf[0] /= sum;
        cmf[1] /= sum;

        for (auto& pt : points)
        {
            // Sample the PMF to determine which face we're creating a splat point on
            vec4 xi = rng.Rand4();
            const int faceIdx = (xi.z < cmf[0]) ? 0 : ((xi.z < cmf[1]) ? 1 : 2);

            vec3 p;
            p[faceIdx] = m_params.dims[faceIdx] * ((xi.w < 0.5f) ? 0.5 : -0.5f);
            p[(faceIdx + 1) % 3] = mix(-1.f, 1.f, xi.x) * m_params.dims[(faceIdx + 1) % 3] * 0.5f;
            p[(faceIdx + 2) % 3] = mix(-1.f, 1.f, xi.y) * m_params.dims[(faceIdx + 2) % 3] * 0.5f;

            // Create the splat
            pt = GenerateRandomGaussianPoint(Tracable::m_params.transform.PointToWorldSpace(p), gaussSigma, rng);
        }

        return points;
    }

    template<>
    __host__ float Host::Primitive<PlaneParams>::CalculateSurfaceArea() const
    {
        // NOTE: Unbounded planes have infinite surface area
        return m_params.isBounded ? sqr(Tracable::m_params.transform.sca) : std::numeric_limits<float>::infinity();
    }

    template<>
    __host__ float Host::Primitive<UnitSphereParams>::CalculateSurfaceArea() const
    {
        return 4. * kPi * sqr(Tracable::m_params.transform.sca);
    }

    template<>
    __host__ float Host::Primitive<CylinderParams>::CalculateSurfaceArea() const
    {
        return 2 * kPi * (1 + m_params.height) * sqr(Tracable::m_params.transform.sca);
    }

    template<>
    __host__ float Host::Primitive<BoxParams>::CalculateSurfaceArea() const
    {
        float area = 0.f;
        for (int d = 0; d < 3; ++d)
        {
            area += 2 * m_params.dims[d] * m_params.dims[(d + 1) % 3];
        }
        return area * sqr(Tracable::m_params.transform.sca);
    }
}