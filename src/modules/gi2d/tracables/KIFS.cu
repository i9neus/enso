#include "KIFS.cuh"

#include "../GenericIntersector.cuh"
#include "../GenericSDF.cuh"
#include "core/math/ColourUtils.cuh"
#include "../bih/BIH2DAsset.cuh"
#include "io/json/JsonUtils.h"

namespace Enso
{
    __host__ __device__ KIFSParams::KIFSParams()        
    {
        m_kifs.pivot = vec2(0.0);        
        m_kifs.isosurface = 0.05f;
        m_kifs.iterScale = 2.0f;
        m_kifs.numIterations = 0;
        m_kifs.rotate = 0.0f;
        m_kifs.objectBounds = 0.2f;
        m_kifs.sdfScale = 10.0f;

        m_intersector.maxIterations = 50;
        m_intersector.cutoffThreshold = 1e-5f;
        m_intersector.escapeThreshold = 1.0f;
        m_intersector.rayIncrement = 0.9f;
        m_intersector.rayKickoff = 1e-4f;
        m_intersector.failThreshold = 1e-2f;
    }

    __host__ __device__ Device::KIFS::KIFS() :
        m_kBary(vec2(1.0, -0.5 / 0.866025f), vec2(0.0f, 1. / 0.866025)),
        m_kBaryInv(vec2(1.0, 0.5f), vec2(0.f, 0.866025)),
        m_rot1(mat2::Identity()),
        m_rot2(mat2::Identity())
    {
    }

    __device__ void Device::KIFS::OnSynchronise(const int flags)
    {
        m_rot1 = mat2(vec2(cos(m_kifs.rotate), -sin(m_kifs.rotate)), vec2(sin(m_kifs.rotate), cos(m_kifs.rotate)));
        m_rot2 = mat2(vec2(1.0, 0.0), vec2(0.0, 1.0));
    }
    
    __host__ __device__ __forceinline__ vec3 Device::KIFS::EvaluateSDF(vec2 z, const mat2& basis) const
    { 
        z *= m_kifs.sdfScale;
        
        // Fold and transform
        constexpr int kMaxIterations = 5;
        uint code = 0u;
        mat2 bi = basis;
        float gamma;
        for (int idx = 0; idx < kMaxIterations && idx < m_kifs.numIterations; ++idx)
        {
            z = m_rot1 * z + m_kifs.pivot;
            bi[0] = m_rot1 * bi[0];
            bi[1] = m_rot1 * bi[1];

            vec2 b = m_kBary * z;
            vec2 bu = (m_kBary * bi[0]) + b;
            vec2 bv = (m_kBary * bi[1]) + b;

            gamma = 1.0f - (b.x + b.y);
            if (gamma > 0.5f)
            {
                b = vec2(0.5f - b.y, 0.5f - b.x);
                bu = vec2(0.5f - bu.y, 0.5f - bu.x);
                bv = vec2(0.5f - bv.y, 0.5f - bv.x);
            }
            if (b.x > 0.5f)
            {
                b = vec2(1.0f - b.x, b.y + b.x - 0.5f);
                bu = vec2(1.0f - bu.x, bu.y + bu.x - 0.5f);
                bv = vec2(1.0f - bv.x, bv.y + bv.x - 0.5f);           
                code = code | (1u << (idx << 1));
            }
            if (b.y > 0.5f)
            {
                b = vec2(b.x + b.y - 0.5f, 1.0f - b.y);
                bu = vec2(bu.x + bu.y - 0.5f, 1.0f - bu.y);
                bv = vec2(bv.x + bv.y - 0.5f, 1.0f - bv.y);
                code = code | (2u << (idx << 1));
            }

            z = (m_kBaryInv * b) * m_kifs.iterScale;
            bi[0] = m_kBaryInv * (bu - b);
            bi[1] = m_kBaryInv * (bv - b);
        }

        // Compute the properties of the field
        vec3 F = SDFLine(z, vec2(-0.5f, 0.0), vec2(1.0f, 0.0)) / m_kifs.sdfScale;
        //vec3 F = SDFPoint(z, vec2(0.0f), 1.0f) / m_kifs.sdfScale;

        // Return the field strength relative to the iso-surface, plus the gradient
        F.x = F.x / powf(m_kifs.iterScale, float(m_kifs.numIterations)) - m_kifs.isosurface;
        
        F.yz = bi * F.yz;
        F.yz = basis[0] * F.y + basis[1] * F.z;
        return F;
    }
    
    __host__ __device__ bool Device::KIFS::IntersectRay(const Ray2D& rayWorld, HitCtx2D& hitWorld) const
    {        
        RayRange2D range;
        if (!IntersectRayBBox(rayWorld, m_worldBBox, range) || range.tNear > hitWorld.tFar) { return false; }

        /*if (hitWorld.debugData)
        {
            KIFSDebugData& debugData = *reinterpret_cast<KIFSDebugData*>(hitWorld.debugData);
            debugData.pNear = rayWorld.PointAt(range.tNear);
            debugData.pFar = rayWorld.PointAt(range.tFar);
            debugData.isHit = false;
        }*/

        RayBasic2D rayObject = RayToObjectSpace(rayWorld);
        HitCtx2D hitObject;        

        float t = fmaxf(0.0f, range.tNear);
    
        // TODO: Adjust this when scaling is enabled
        const float localMag = 1.0f;
        //rayObject.d /= localMag;
        //t *= localMag;
        //rayObject.o *= m_kifs.sdfScale;

        // The transpose of the basis of the ray tangent and its direction
        mat2 basis(-rayObject.d.y, rayObject.d.x, rayObject.d.x, rayObject.d.y);
        vec2 p = rayObject.PointAt(t);
        vec3 F;
        int iterIdx = 0;
        constexpr int kMaxIterations = 50;
        bool isSubsurface = false;
        for (iterIdx = 0; iterIdx < kMaxIterations && iterIdx < m_intersector.maxIterations; iterIdx++)
        {
            F = EvaluateSDF(p, basis);

            /*if (hitWorld.debugData && iterIdx < KIFSDebugData::kMaxPoints)
            {
                KIFSDebugData& debugData = *reinterpret_cast<KIFSDebugData*>(hitWorld.debugData);
                debugData.marchPts[iterIdx] = PointToWorldSpace(p);
            }*/

            // On the first iteration, simply determine whether we're inside the isosurface or not
            if (iterIdx == 0) { isSubsurface = F.x < 0.0; }
            // Otherwise, check to see if we're at the surface
            else if (fabsf(F.x) < m_intersector.cutoffThreshold * localMag) { break; }

            //if (F.x > m_intersector.escapeThreshold) { hitWorld.AccumDebug(kBlue); return false; }

            // Increment the ray position based on the SDF magnitude
            t += (isSubsurface ? -F.x : F.x) * m_intersector.rayIncrement;

            // If we've gone beyond t-near, bail out now
            if (t / localMag > hitWorld.tFar) { return false; }

            p = rayObject.PointAt(t);
        }

        if (F.x > m_intersector.failThreshold) { return false; }
        
        t /= localMag;

        if (t >= hitWorld.tFar) { return false; }
        
        hitWorld.tFar = t;
        // TODO: Transform normals into screen space
        hitWorld.n = normalize(F.yz);
        hitWorld.kickoff = m_intersector.rayKickoff;

        /*if (hitWorld.debugData)
        {
            KIFSDebugData& debugData = *reinterpret_cast<KIFSDebugData*>(hitWorld.debugData);
            debugData.normal = hitWorld.n;
            debugData.hit = PointToWorldSpace(p);
            debugData.isHit = true;
        }*/

        return true;        
    }

    __host__ __device__ bool Device::KIFS::Contains(const UIViewCtx& viewCtx) const
    {
        if (!m_worldBBox.Contains(viewCtx.mousePos)) { return false; }
        uint code;
        return EvaluateSDF(viewCtx.mousePos - m_transform.trans, mat2(1.0f, 0.0f, 0.0f, 1.0f), code).x < 0.0f;
    }

    __device__ vec4 Device::KIFS::EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx) const
    {
        vec2 pObject = pWorld - m_transform.trans;

        vec3 F = EvaluateSDF(pObject, mat2(1.0f, 0.0f, 0.0f, 1.0f));

        return (F.x < 0.0f) ? vec4(kOne, 1.0f) : vec4(0.0f);
        //return vec4(vec3(normalize(F.yz) + 1.0f * 0.5f, 0.0f), 1.0f);
    }

    __host__ Host::KIFS::KIFS(const std::string& id) :
        Tracable(id, m_hostInstance),
        cu_deviceInstance(nullptr)
    {
        SetAttributeFlags(kSceneObjectInteractiveElement);
        cu_deviceInstance = InstantiateOnDevice<Device::KIFS>();
        Synchronise(kSyncObjects);
    }

    __host__ AssetHandle<Host::GenericObject> Host::KIFS::Instantiate(const std::string& id, const Json::Node&)
    {
        return CreateAsset<Host::KIFS>(id);
    }

    __host__ Host::KIFS::~KIFS()
    {
        OnDestroyAsset();
    }

    __host__ void Host::KIFS::OnDestroyAsset()
    {
        DestroyOnDevice(cu_deviceInstance);
    }

    __host__ void Host::KIFS::Synchronise(const int syncType)
    {
        Tracable::Synchronise(cu_deviceInstance, syncType);

        if (syncType & kSyncParams)
        {
            SynchroniseInheritedClass<KIFSParams>(cu_deviceInstance, m_hostInstance, kSyncParams);
        }
    }

    __host__ uint Host::KIFS::OnCreate(const std::string& stateID, const UIViewCtx& viewCtx)
    {
        const vec2 mousePosLocal = viewCtx.mousePos - m_transform.trans;
        if (stateID == "kCreateSceneObjectOpen" || stateID == "kCreateSceneObjectHover")
        {
            m_transform.trans = viewCtx.mousePos;
            m_isConstructed = true;
            SetDirtyFlags(kDirtyObjectBounds);

            if (stateID == "kCreateSceneObjectOpen") { Log::Success("Opened KIFS %s", GetAssetID()); }
        }
        else if (stateID == "kCreateSceneObjectAppend")
        {
            m_isFinalised = true;
            SetDirtyFlags(kDirtyObjectBounds);
        }

        return m_dirtyFlags;
    }

    __host__ bool Host::KIFS::Rebuild(const uint parentFlags, const UIViewCtx& viewCtx)
    {
        if (!m_dirtyFlags) { return IsConstructed(); }

        bool resyncParams = false;
        if (m_dirtyFlags & kDirtyObjectBounds)
        {
            RecomputeBoundingBoxes();
            resyncParams = true;
        }
        if (m_dirtyFlags & kDirtyMaterials)
        {
            resyncParams = true;
        }

        if (resyncParams) { Synchronise(kSyncParams); }

        ClearDirtyFlags();
        return IsConstructed();
    }

    __host__ bool Host::KIFS::Contains(const UIViewCtx& viewCtx) const
    {
        return m_hostInstance.Contains(viewCtx);
    }

    __host__ BBox2f Host::KIFS::RecomputeObjectSpaceBoundingBox()
    {
        return BBox2f(vec2(-m_hostInstance.m_kifs.objectBounds), vec2(m_hostInstance.m_kifs.objectBounds));
    }

    __host__ bool Host::KIFS::Serialise(Json::Node& node, const int flags) const
    {
        Tracable::Serialise(node, flags);

        Json::Node kifsNode = node.AddChildObject("kifs");
        kifsNode.AddValue("rotate", m_hostInstance.m_kifs.rotate);
        kifsNode.AddVector("pivot", m_hostInstance.m_kifs.pivot);
        kifsNode.AddValue("isosurface", m_hostInstance.m_kifs.isosurface);
        kifsNode.AddValue("kifsIterations", m_hostInstance.m_kifs.numIterations);
        kifsNode.AddValue("scale", m_hostInstance.m_kifs.sdfScale);

        Json::Node isectNode = node.AddChildObject("isector");
        isectNode.AddValue("cutoffThreshold", m_hostInstance.m_intersector.cutoffThreshold);
        isectNode.AddValue("escapeThreshold", m_hostInstance.m_intersector.escapeThreshold);
        isectNode.AddValue("failThreshold", m_hostInstance.m_intersector.failThreshold);
        isectNode.AddValue("rayIncrement", m_hostInstance.m_intersector.rayIncrement);
        isectNode.AddValue("rayKickoff", m_hostInstance.m_intersector.rayKickoff);
        isectNode.AddValue("maxIterations", m_hostInstance.m_intersector.maxIterations);

        return true;
    }

    __host__ uint Host::KIFS::Deserialise(const Json::Node& node, const int flags)
    {
        Tracable::Deserialise(node, flags);

        Json::Node kifsNode = node.GetChildObject("kifs", flags);
        bool dirty = false;
        if (kifsNode)
        {
            dirty |= kifsNode.GetValue("rotate", m_hostInstance.m_kifs.rotate, flags);
            dirty |= kifsNode.GetVector("pivot", m_hostInstance.m_kifs.pivot, flags);
            dirty |= kifsNode.GetValue("isosurface", m_hostInstance.m_kifs.isosurface, flags);
            dirty |= kifsNode.GetValue("iterations", m_hostInstance.m_kifs.numIterations, flags);
            dirty |= kifsNode.GetValue("scale", m_hostInstance.m_kifs.sdfScale, flags);
        }

        Json::Node isectNode = node.GetChildObject("isector", flags);
        if (isectNode)
        {
            dirty |= isectNode.GetValue("cutoffThreshold", m_hostInstance.m_intersector.cutoffThreshold, flags);
            dirty |= isectNode.GetValue("escapeThreshold", m_hostInstance.m_intersector.escapeThreshold, flags);
            dirty |= isectNode.GetValue("failThreshold", m_hostInstance.m_intersector.failThreshold, flags);
            dirty |= isectNode.GetValue("rayIncrement", m_hostInstance.m_intersector.rayIncrement, flags);
            dirty |= isectNode.GetValue("rayKickoff", m_hostInstance.m_intersector.rayKickoff, flags);
            dirty |= isectNode.GetValue("maxIterations", m_hostInstance.m_intersector.maxIterations, flags);
        }
        
        if (dirty) { SetDirtyFlags(kDirtyMaterials); }
        return m_dirtyFlags;
    }
}