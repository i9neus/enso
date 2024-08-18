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
                hit.matID = Tracable::m_params.materialIdx;
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
            hit.matID = Tracable::m_params.materialIdx;

            return true;
        }
    }

    template<typename ParamsType>
    __host__ Host::Primitive<ParamsType>::Primitive(const InitCtx& initCtx) :
        Tracable(initCtx),
        cu_deviceInstance(AssetAllocator::InstantiateOnDevice<DeviceType>(*this))
    {
        Tracable::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::Tracable>(cu_deviceInstance));
    }

    template<typename ParamsType>
    __host__ Host::Primitive<ParamsType>::Primitive(const InitCtx& initCtx, const BidirectionalTransform& transform, const int materialIdx, const ParamsType& params) :
        Primitive(initCtx)
    {
        m_params = params;
        Tracable::m_params.transform = transform;
        Tracable::m_params.materialIdx = materialIdx;
        Synchronise(kSyncParams);
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
    __host__ std::vector<GaussianPoint> Host::Primitive<PlaneParams>::GenerateGaussianPointCloud(const int numPoints, MersenneTwister& rng)
    {
        if (!m_params.isBounded)
        {
            Log::Error("Warning: cannot create points on an unbounded plane!");
            return std::vector<GaussianPoint>();
        }

        const float gaussSigma = 0.5f * std::sqrt(CalculateSurfaceArea() / numPoints);

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
    __host__ std::vector<GaussianPoint> Host::Primitive<UnitSphereParams>::GenerateGaussianPointCloud(const int numPoints, MersenneTwister& rng)
    {
        const float gaussSigma = 0.5f * std::sqrt(CalculateSurfaceArea() / numPoints);

        std::vector<GaussianPoint> points(numPoints);
        for (auto& pt : points)
        {
            const vec3 p = SampleUnitSphere(rng.Rand2());
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
    
    template<typename ParamsType>
    __host__ Host::Primitive<ParamsType>::~Primitive()
    {
        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);
    }
}