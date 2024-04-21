#include "KIFS.cuh"

#include "../primitives/GenericIntersector.cuh"
#include "../primitives/SDF.cuh"
#include "core/math/ColourUtils.cuh"
#include "../bih/BIH2DAsset.cuh"
#include "io/json/JsonUtils.h"

namespace Enso
{
    __host__ __device__ KIFSParams::KIFSParams()        
    {
        kifs.pivot = vec2(0.0);        
        kifs.isosurface = 0.0038f;
        kifs.iterScale = 2.0f;
        kifs.numIterations = 4;
        kifs.rotate = 0.944f;
        kifs.objectBounds = 0.5f;
        kifs.sdfScale = 4.5f;
        kifs.primSize = 17.0f;

        intersector.maxIterations = 50;
        intersector.cutoffThreshold = 1e-5f;
        intersector.escapeThreshold = 1.0f;
        intersector.rayIncrement = 0.9f;
        intersector.rayKickoff = 1e-4f;
        intersector.failThreshold = 1e-2f;

        look.phase = 0.0f;
        look.range = 1.0f;
    }

    __host__ __device__ Device::KIFS::KIFS() :
        m_kBary(vec2(1.0, -0.5 / 0.866025f), vec2(0.0f, 1. / 0.866025)),
        m_kBaryInv(vec2(1.0, 0.5f), vec2(0.f, 0.866025)),
        m_rot1(mat2::Identity()),
        m_rot2(mat2::Identity())
    {
    }

    __host__ __device__ void Device::KIFS::Synchronise(const KIFSParams& params) 
    {  
        m_params = params; 

        m_rot1 = mat2(vec2(cos(m_params.kifs.rotate), -sin(m_params.kifs.rotate)), vec2(sin(m_params.kifs.rotate), cos(m_params.kifs.rotate)));
        m_rot2 = mat2(vec2(1.0, 0.0), vec2(0.0, 1.0));
    }
    
    __host__ __device__ __forceinline__ vec3 Device::KIFS::EvaluateSDF(vec2 z, const mat2& basis, uint& code) const
    { 
        z *= m_params.kifs.sdfScale;
        
        // Fold and transform
        constexpr int kMaxIterations = 5;
        code = 0u;
        mat2 bi = basis;
        float gamma;
        for (int idx = 0; idx < kMaxIterations && idx < m_params.kifs.numIterations; ++idx)
        {
            z = m_rot1 * z + m_params.kifs.pivot;
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

            z = (m_kBaryInv * b) * m_params.kifs.iterScale;
            bi[0] = m_kBaryInv * (bu - b);
            bi[1] = m_kBaryInv * (bv - b);
        }

        // Compute the properties of the field
        vec3 F = SDF::Line(z, vec2(-0.5f * m_params.kifs.primSize, 0.0), vec2(1.0f * m_params.kifs.primSize, 0.0)) / m_params.kifs.sdfScale;
        //vec3 F = SDFPoint(z, vec2(0.0f), 1.0f) / m_params.kifs.sdfScale;

        // Return the field strength relative to the iso-surface, plus the gradient
        F.x = F.x / powf(m_params.kifs.iterScale, float(m_params.kifs.numIterations)) - m_params.kifs.isosurface;
        
        F.yz = bi * F.yz;
        F.yz = basis[0] * F.y + basis[1] * F.z;
        return F;
    }
    
    __host__ __device__ bool Device::KIFS::IntersectRay(const Ray2D& rayWorld, HitCtx2D& hitWorld) const
    {        
        RayRange2D range;
        if (!IntersectRayBBox(rayWorld, GetWorldBBox(), range) || range.tNear > hitWorld.tFar) { return false; }

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
        //rayObject.o *= m_params.kifs.sdfScale;

        // The transpose of the basis of the ray tangent and its direction
        mat2 basis(-rayObject.d.y, rayObject.d.x, rayObject.d.x, rayObject.d.y);
        vec2 p = rayObject.PointAt(t);
        vec3 F;
        uint code;
        int iterIdx = 0;
        constexpr int kMaxIterations = 50;
        bool isSubsurface = false;
        for (iterIdx = 0; iterIdx < kMaxIterations && iterIdx < m_params.intersector.maxIterations; iterIdx++)
        {
            F = EvaluateSDF(p, basis, code);

            /*if (hitWorld.debugData && iterIdx < KIFSDebugData::kMaxPoints)
            {
                KIFSDebugData& debugData = *reinterpret_cast<KIFSDebugData*>(hitWorld.debugData);
                debugData.marchPts[iterIdx] = PointToWorldSpace(p);
            }*/

            // On the first iteration, simply determine whether we're inside the isosurface or not
            if (iterIdx == 0) { isSubsurface = F.x < 0.0; }
            // Otherwise, check to see if we're at the surface
            else if (fabsf(F.x) < m_params.intersector.cutoffThreshold * localMag) { break; }

            //if (F.x > intersector.escapeThreshold) { hitWorld.AccumDebug(kBlue); return false; }

            // Increment the ray position based on the SDF magnitude
            t += (isSubsurface ? -F.x : F.x) * m_params.intersector.rayIncrement;

            // If we've left the bounding box, bail out now
            if (t / localMag > hitWorld.tFar) { return false; }

            p = rayObject.PointAt(t);
        }

        if (F.x > m_params.intersector.failThreshold) { return false; }
        
        t /= localMag;

        if (t >= hitWorld.tFar) { return false; }
        
        hitWorld.tFar = t;
        // TODO: Transform normals into screen space
        hitWorld.n = normalize(F.yz);
        hitWorld.kickoff = m_params.intersector.rayKickoff;
        hitWorld.hash = uint(fract(IntToUnitFloat(HashOf(code)) * m_params.look.range + m_params.look.phase) * float(0xffffffff));

        /*if (hitWorld.debugData)
        {
            KIFSDebugData& debugData = *reinterpret_cast<KIFSDebugData*>(hitWorld.debugData);
            debugData.normal = hitWorld.n;
            debugData.hit = PointToWorldSpace(p);
            debugData.isHit = true;
        }*/

        return true;        
    }

    __host__ __device__ uint Device::KIFS::OnMouseClick(const UIViewCtx& viewCtx) const
    {
        if (!GetWorldBBox().Contains(viewCtx.mousePos)) { return false; }
        uint code;
        return (EvaluateSDF(viewCtx.mousePos - GetTransform().trans, mat2(1.0f, 0.0f, 0.0f, 1.0f), code).x < 0.0f) ? kSceneObjectPrecisionDrag : kSceneObjectInvalidSelect;
    }

    __host__ __device__ vec4 Device::KIFS::EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx, const bool isMouseTest) const
    {
        if (!GetWorldBBox().Contains(pWorld)) { return vec4(0.0f); }
        
        vec2 pObject = pWorld - GetTransform().trans;
        uint code;
        vec3 F = EvaluateSDF(pObject, mat2(1.0f, 0.0f, 0.0f, 1.0f), code);

        return (F.x < 0.0f) ? vec4(Hue(fract(IntToUnitFloat(HashOf(code)) * m_params.look.range + m_params.look.phase)), 1.0f) : vec4(0.0f);
        //return vec4(vec3(normalize(F.yz) + 1.0f * 0.5f, 0.0f), 1.0f);
    }

    __host__ Host::KIFS::KIFS(const Asset::InitCtx& initCtx) :
        Tracable(initCtx, m_hostInstance, nullptr),
        cu_deviceInstance(AssetAllocator::InstantiateOnDevice<Device::KIFS>(*this))
    {
        SetAttributeFlags(kSceneObjectInteractiveElement);
        Tracable::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::Tracable>(cu_deviceInstance));

        Synchronise(kSyncObjects);
    }

    __host__ AssetHandle<Host::GenericObject> Host::KIFS::Instantiate(const std::string& id, const Json::Node&, const AssetHandle<const Host::SceneContainer>&)
    {
        return AssetAllocator::CreateAsset<Host::KIFS>(id);
    }

    __host__ Host::KIFS::~KIFS() noexcept
    {
        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);
    }

    __host__ void Host::KIFS::Synchronise(const uint syncType)
    {
        Tracable::Synchronise(syncType);

        if (syncType & kSyncParams)
        {
            SynchroniseObjects<Device::KIFS>(cu_deviceInstance, m_hostInstance.m_params);
            m_hostInstance.Synchronise(m_hostInstance.m_params);
        }
    }

    __host__ bool Host::KIFS::OnCreate(const std::string& stateID, const UIViewCtx& viewCtx)
    {
        const vec2 mousePosLocal = viewCtx.mousePos - GetTransform().trans;
        if (stateID == "kCreateSceneObjectOpen" || stateID == "kCreateSceneObjectHover")
        {
            GetTransform().trans = viewCtx.mousePos;
            m_isConstructed = true;
            SignalDirty({ kDirtyObjectBoundingBox, kDirtyObjectRebuild });

            if (stateID == "kCreateSceneObjectOpen") { Log::Success("Opened KIFS %s", GetAssetID()); }
        }
        else if (stateID == "kCreateSceneObjectAppend")
        {
            m_isFinalised = true;
            SignalDirty({ kDirtyObjectBoundingBox, kDirtyObjectRebuild });
        }

        return true;
    }

    __host__ bool Host::KIFS::Rebuild()
    {
        RecomputeBoundingBoxes();        
        Synchronise(kSyncParams); 
        
        return true;
    }

    __host__ uint Host::KIFS::OnMouseClick(const UIViewCtx& viewCtx) const
    {
        return m_hostInstance.OnMouseClick(viewCtx);
    }

    __host__ BBox2f Host::KIFS::RecomputeObjectSpaceBoundingBox()
    {
        return BBox2f(vec2(-m_hostInstance.m_params.kifs.objectBounds), vec2(m_hostInstance.m_params.kifs.objectBounds));
    }

    __host__ bool Host::KIFS::Serialise(Json::Node& node, const int flags) const
    {
        Tracable::Serialise(node, flags);

        Json::Node kifsNode = node.AddChildObject("kifs");
        const auto& kifs = m_hostInstance.m_params.kifs;
        kifsNode.AddValue("rotate", kifs.rotate);
        kifsNode.AddVector("pivot", kifs.pivot);
        kifsNode.AddValue("isosurface", kifs.isosurface);
        kifsNode.AddValue("primSize", kifs.primSize);
        kifsNode.AddValue("iterations", kifs.numIterations);
        kifsNode.AddValue("scale", kifs.sdfScale);

        Json::Node isectNode = node.AddChildObject("isector");
        const auto& isect = m_hostInstance.m_params.intersector;
        isectNode.AddValue("cutoffThreshold", isect.cutoffThreshold);
        isectNode.AddValue("escapeThreshold", isect.escapeThreshold);
        isectNode.AddValue("failThreshold", isect.failThreshold);
        isectNode.AddValue("rayIncrement", isect.rayIncrement);
        isectNode.AddValue("rayKickoff", isect.rayKickoff);
        isectNode.AddValue("maxIterations", isect.maxIterations);

        Json::Node lookNode = node.AddChildObject("look");
        lookNode.AddValue("phase", m_hostInstance.m_params.look.phase);
        lookNode.AddValue("range", m_hostInstance.m_params.look.range);

        return true;
    }

    __host__ bool Host::KIFS::Deserialise(const Json::Node& node, const int flags)
    {
        bool isDirty = Tracable::Deserialise(node, flags);

        Json::Node kifsNode = node.GetChildObject("kifs", flags);        
        if (kifsNode)
        {
            auto& kifs = m_hostInstance.m_params.kifs;
            isDirty |= kifsNode.GetValue("rotate", kifs.rotate, flags);
            isDirty |= kifsNode.GetVector("pivot", kifs.pivot, flags);
            isDirty |= kifsNode.GetValue("isosurface", kifs.isosurface, flags);
            isDirty |= kifsNode.GetValue("primSize", kifs.primSize, flags);
            isDirty |= kifsNode.GetValue("iterations", kifs.numIterations, flags);
            isDirty |= kifsNode.GetValue("scale", kifs.sdfScale, flags);
        }

        Json::Node isectNode = node.GetChildObject("isect", flags);
        if (isectNode)
        {
            auto& isect = m_hostInstance.m_params.intersector;
            isDirty |= isectNode.GetValue("cutoffThreshold", isect.cutoffThreshold, flags);
            isDirty |= isectNode.GetValue("escapeThreshold", isect.escapeThreshold, flags);
            isDirty |= isectNode.GetValue("failThreshold", isect.failThreshold, flags);
            isDirty |= isectNode.GetValue("rayIncrement", isect.rayIncrement, flags);
            isDirty |= isectNode.GetValue("rayKickoff", isect.rayKickoff, flags);
            isDirty |= isectNode.GetValue("maxIterations", isect.maxIterations, flags);
        }

        Json::Node lookNode = node.GetChildObject("look", flags);
        if (lookNode)
        {
            isDirty |= lookNode.GetValue("phase", m_hostInstance.m_params.look.phase, flags);
            isDirty |= lookNode.GetValue("range", m_hostInstance.m_params.look.range, flags);
        }
        
        if (isDirty) 
        { 
            SignalDirty({ kDirtyParams, kDirtyIntegrators }); 
        }

        return isDirty;
    }
}