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
        //m_kifs.objectBounds = BBox2f(vec2(-0.1), vec2(0.1));
        m_kifs.objectBounds = 0.2f;
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
    
    __host__ __device__ __forceinline__ bool  Device::KIFS::SampleSDF(vec2 z, vec3& F) const
    { 
        if (length2(z) > m_kifs.objectBounds) { return false; }

        z *= 10.f;

        // Fold and transform
        constexpr int kMaxIterations = 5;
        uint code = 0u;
        vec2 bary;
        float gamma;
        for (int idx = 0; idx < kMaxIterations && idx < m_kifs.numIterations; ++idx)
        {
            z = m_rot1 * z;
            z += m_kifs.pivot;
            z = m_rot2 * z;

            bary = m_kBary * z;
            gamma = 1.0f - (bary.x + bary.y);

            if (gamma > 0.5)
            {
                bary = vec2(0.5 - bary.y, 0.5 - bary.x);
            }
            if (bary.x > 0.5)
            {
                bary = vec2(1. - bary.x, bary.y + bary.x - 0.5);
                code = code | (1u << (idx << 1));
            }
            if (bary.y > 0.5)
            {
                bary = vec2(bary.x + bary.y - 0.5, 1. - bary.y);
                code = code | (2u << (idx << 1));
            }

            z = ((m_kBaryInv * bary) - m_kifs.pivot) * m_kifs.iterScale;
        }

        //uint mask = HashOf(code);  
        vec3 sdf = SDFLine(z, vec2(-0.5f, 0.0), vec2(1.0f, 0.0));

        F.x = sdf.x / powf(m_kifs.iterScale, float(m_kifs.numIterations)) - m_kifs.isosurface;
        F.yz = ((F.x >= 0.0f) ? sdf.yz : -sdf.yz) / sdf.x;
        return true;
    }
    
    __host__ __device__ bool Device::KIFS::IntersectRay(const Ray2D& rayWorld, HitCtx2D& hitWorld) const
    {
        return false;
    }

    __device__ vec4 Device::KIFS::EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx) const
    {
        const vec2 pObject = pWorld - m_transform.trans;        
        vec3 F;
        if (!Iterate(pObject, F)) { return vec4(0.0f); }

        return (F.x < 0.0f) ? vec4(kOne, 1.0f) : vec4(0.0f);
        //return vec4(vec3(F.x), 1.0f);
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

        Device::KIFS test;
        vec3 F;
        test.m_kifs.numIterations = 1;
        test.m_kifs.isosurface = 0.0f;
        test.Iterate(vec2(0.0f), F);

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

    __host__ BBox2f Host::KIFS::RecomputeObjectSpaceBoundingBox()
    {
        return BBox2f(vec2(-m_hostInstance.m_kifs.objectBounds), vec2(m_hostInstance.m_kifs.objectBounds));
    }

    __host__ bool Host::KIFS::Serialise(Json::Node& node, const int flags) const
    {
        Tracable::Serialise(node, flags);

        node.AddValue("rotate", m_hostInstance.m_kifs.rotate);
        node.AddVector("pivot", m_hostInstance.m_kifs.pivot);
        node.AddValue("isosurface", m_hostInstance.m_kifs.isosurface);
        node.AddValue("iterations", m_hostInstance.m_kifs.numIterations);

        return true;
    }

    __host__ uint Host::KIFS::Deserialise(const Json::Node& node, const int flags)
    {
        Tracable::Deserialise(node, flags);

        if (node.GetValue("rotate", m_hostInstance.m_kifs.rotate, flags)) { SetDirtyFlags(kDirtyMaterials); }
        if (node.GetVector("pivot", m_hostInstance.m_kifs.pivot, flags)) { SetDirtyFlags(kDirtyMaterials); }
        if (node.GetValue("isosurface", m_hostInstance.m_kifs.isosurface, flags)) { SetDirtyFlags(kDirtyMaterials); }
        if (node.GetValue("iterations", m_hostInstance.m_kifs.numIterations, flags)) { SetDirtyFlags(kDirtyMaterials); }

        return m_dirtyFlags;
    }
}