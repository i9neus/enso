#include "CudaSDF.cuh"
#include "generic/JsonUtils.h"

#include <random>

namespace Cuda
{
    __host__ __device__ SDFParams::SDFParams()
    {
        maxSpecularIterations = 50;
        maxDiffuseIterations = 15;
        cutoffThreshold = 1e-4f;
        escapeThreshold = 1.0f;
        rayIncrement = 0.9f;
        rayKickoff = 1e-4f;
        failThreshold = 1e-2f;

        sphere.r = 0.5f;
        torus.r1 = 0.4f;
        torus.r2 = 0.1f;
        box.size = 0.5f;
    }

    __host__ SDFParams::SDFParams(const ::Json::Node& node, const uint flags) :
        SDFParams()
    {
        FromJson(node, flags);
    }

    __host__ void SDFParams::Update(const uint operation)
    {     
        transform.Update(operation);
    }

    __host__ void SDFParams::ToJson(::Json::Node& node) const
    {      
        node.AddEnumeratedParameter("primitiveType", std::vector<std::string>({ "sphere", "torus", "box", "capsule" }), primitiveType);     
        node.AddValue("maxSpecularIterations", maxSpecularIterations);
        node.AddValue("maxDiffuseIterations", maxDiffuseIterations);
        node.AddValue("cutoffThreshold", cutoffThreshold);
        node.AddValue("escapeThreshold", escapeThreshold);
        node.AddValue("rayIncrement", rayIncrement);
        node.AddValue("rayKickoff", rayKickoff);
        node.AddValue("failThreshold", failThreshold);

        Json::Node sphereNode = node.AddChildObject("sphere");
        sphereNode.AddValue("r", sphere.r);

        Json::Node torusNode = node.AddChildObject("torus");
        torusNode.AddValue("r1", torus.r1);
        torusNode.AddValue("r2", torus.r2);

        Json::Node boxNode = node.AddChildObject("box");
        boxNode.AddValue("size", box.size);

        tracable.ToJson(node);
        transform.ToJson(node);
    }

    __host__ uint SDFParams::FromJson(const ::Json::Node& node, const uint flags)
    {
        node.GetEnumeratedParameter("primitiveType", std::vector<std::string>({ "sphere", "torus", "box", "capsule" }), primitiveType, flags);
        node.GetValue("maxSpecularIterations", maxSpecularIterations, Json::kSilent);
        node.GetValue("maxDiffuseIterations", maxDiffuseIterations, Json::kSilent);
        node.GetValue("cutoffThreshold", cutoffThreshold, Json::kSilent);
        node.GetValue("escapeThreshold", escapeThreshold, Json::kSilent);
        node.GetValue("rayIncrement", rayIncrement, Json::kSilent);
        node.GetValue("rayKickoff", rayKickoff, Json::kSilent);
        node.GetValue("failThreshold", failThreshold, Json::kSilent);

        Json::Node sphereNode = node.GetChildObject("sphere", Json::kSilent);
        if (sphereNode) { sphereNode.GetValue("r", sphere.r, flags); }

        Json::Node torusNode = node.GetChildObject("torus", Json::kSilent);
        if (torusNode) 
        { 
            torusNode.GetValue("r1", torus.r1, flags);
            torusNode.GetValue("r2", torus.r2, flags);
        }

        Json::Node boxNode = node.GetChildObject("box", Json::kSilent);
        if (boxNode) { boxNode.GetValue("size", box.size, flags); }
        
        tracable.FromJson(node, flags);
        transform.FromJson(node, flags);

        return kRenderObjectDirtyAll;
    }

    __device__ Device::SDF::SDF() { }

    __device__  bool Device::SDF::Intersect(Ray& globalRay, HitCtx& hitCtx) const
    {
        if (globalRay.flags & kRayLightProbe && m_params.tracable.renderObject.flags() & kRenderObjectExcludeFromBake) { return false; }

        RayBasic localRay = RayToObjectSpace(globalRay.od, m_params.transform);

        float t = Intersector::RayBox(localRay, m_params.escapeThreshold / kRoot2);
        if (t == kNoIntersect) { return false; }

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
        const int maxIterations = (!(globalRay.flags & kRayScattered)) ? m_params.maxSpecularIterations : m_params.maxDiffuseIterations;

        for (i = 0; i < maxIterations; i++)
        {
            //Test the signed distance function
            switch (m_params.primitiveType)
            {
            case kSDFPrimitiveSphere:
                F = SDFPrimitive::Sphere(p, m_params.sphere.r); break;
            case kSDFPrimitiveTorus:
                F = SDFPrimitive::Torus(p, m_params.torus.r1, m_params.torus.r2); break;
            default:
                F = SDFPrimitive::Box(p, m_params.box.size);
            }

            // On the first iteration, simply determine whether we're inside the isosurface or not
            if (i == 0) { isSubsurface = F.x < 0.0; }
            // Otherwise, check to see if we're at the surface
            else if (F.x > 0.0 && F.x < m_params.cutoffThreshold * localMag) { hitCtx.debug = vec3(0.0f, 0.0f, 1.0f) * mix(0.5f, 1.0f, float(i) / float(maxIterations)); break; }

            if (F.x > m_params.escapeThreshold) { hitCtx.debug = vec3(1.0, 0.0f, 0.0f) * mix(0.5f, 1.0f, float(i) / float(maxIterations)); return false; }

            // Increment the ray position based on the SDF magnitude
            t += (isSubsurface ? -F.x : F.x) * m_params.rayIncrement;

            // If we've gone beyond t-near, bail out now
            if (t / localMag > globalRay.tNear) { hitCtx.debug = vec3(1.0, 1.0f, 0.0f) * mix(0.5f, 1.0f, float(i) / float(maxIterations)); return false; }

            p = localRay.PointAt(t);
        }

        if (F.x > m_params.failThreshold) { hitCtx.debug = vec3(0.0f, 1.0f, 0.0f) * mix(0.5f, 1.0f, float(i) / float(maxIterations)); return false; }
        t /= localMag;
        hitCtx.debug *= F.x;

        globalRay.tNear = t;
        hitCtx.Set(HitPoint(globalRay.HitPoint(), NormalToWorldSpace(F.yzw, m_params.transform)),
            isSubsurface,
            vec2(*reinterpret_cast<float*>(&code), 0.0f),  // Dump the bits of the code into the float. FIXME: Not type safe, so fix this
            m_params.rayKickoff,
            kNotALight
        );

        return true;
    }

    __host__ AssetHandle<Host::RenderObject> Host::SDF::Instantiate(const std::string& id, const AssetType& expectedType, const ::Json::Node& json)
    {
        if (expectedType != AssetType::kTracable) { return AssetHandle<Host::RenderObject>(); }

        return CreateAsset<Host::SDF>(id, json);
    }

    __host__  Host::SDF::SDF(const std::string& id, const ::Json::Node& node) :
        Tracable(id),
        cu_deviceData(nullptr)
    {
        cu_deviceData = InstantiateOnDevice<Device::SDF>();
        FromJson(node, ::Json::kSilent);
    }

    __host__ void Host::SDF::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceData);
    }

    __host__ uint Host::SDF::FromJson(const ::Json::Node& node, const uint flags)
    {
        Host::Tracable::FromJson(node, flags);

        m_params.FromJson(node, flags);
        RenderObject::SetUserFacingRenderObjectFlags(m_params.tracable.renderObject.flags());

        SynchroniseObjects(cu_deviceData, m_params);

        return kRenderObjectDirtyAll;
    }
}