#include "SceneObject.cuh"
#include "core/math/ColourUtils.cuh"
#include "io/json/JsonUtils.h"

namespace Enso
{
    __device__ bool Device::SceneObject::EvaluateControlHandles(const vec2& pWorld, const UIViewCtx& viewCtx, vec4& L) const
    {
        // Draw the bounding box
        /*if (m_params.objectBBox.PointOnPerimiter(p, m_params.viewCtx.dPdXY * 2.f))
        {
            L = vec4(kOne, 1.0f);
            return true;
        }*/

        // Draw the control handles
        if (m_handleInnerBBox.IsValid() && !m_handleInnerBBox.Contains(pWorld))
        {
            //L = Blend(L, kOne, 0.2f);

            /*vec2 hp = 2.0f * (p - m_handleInnerBBox.lower) / (m_handleInnerBBox.Dimensions() - vec2(m_params.viewCtx.dPdXY * 10.0f));

            if (fract(hp.x) < 0.1f && fract(hp.y) < 0.1f)
            {
                 L = vec4(1.0f);
                 return true;
            }*/
        }

        return false;
    }

    __host__ Host::SceneObject::SceneObject(const std::string& id, Device::SceneObject& hostInstance, const AssetHandle<const Host::SceneDescription>& scene) :
        GenericObject(id),
        m_hostInstance(hostInstance)
    {
    }

    __host__ void Host::SceneObject::SetDeviceInstance(Device::SceneObject* deviceInstance)
    {
        GenericObject::SetDeviceInstance(m_allocator.StaticCastOnDevice<Device::GenericObject>(deviceInstance));
        cu_deviceInstance = deviceInstance;
    }

    __host__ uint Host::SceneObject::OnSelect(const bool isSelected)
    {
        if (SetGenericFlags(m_hostInstance.m_params.attrFlags, uint(kSceneObjectSelected), isSelected))
        {
            SetDirtyFlags(kDirtyUI, true);
        }
        return m_dirtyFlags;
    }

    __host__ uint Host::SceneObject::OnMove(const std::string& stateID, const UIViewCtx& viewCtx)
    {
        if (stateID == "kMoveSceneObjectBegin")
        {
            m_onMove.dragAnchor = viewCtx.mousePos;
            m_onMove.isDragging = true;
            Log::Error("kMoveSceneObjectBegin");
        }
        else if (stateID == "kMoveSceneObjectDragging")
        {
            Assert(m_onMove.isDragging);

            const vec2 delta = viewCtx.mousePos - m_onMove.dragAnchor;
            m_onMove.dragAnchor = viewCtx.mousePos;
            m_hostInstance.m_params.transform.trans += delta;
            m_hostInstance.m_params.worldBBox += delta;

            // The geometry internal to this object hasn't changed, but it will affect the 
            Log::Warning("kMoveSceneObjectDragging");
            SetDirtyFlags(kDirtyObjectBounds);
        }
        else if (stateID == "kMoveSceneObjectEnd")
        {
            m_onMove.isDragging = false;
            Log::Success("kMoveSceneObjectEnd");
        }

        return m_dirtyFlags;
    }

    __host__ bool Host::SceneObject::Serialise(Json::Node& node, const int flags) const
    {
        Json::Node sceneNode = node.AddChildObject("sceneobject");

        m_hostInstance.m_params.transform.Serialise(sceneNode, flags);
        return true;
    }

    __host__ uint Host::SceneObject::Deserialise(const Json::Node& node, const int flags)
    {
        const Json::Node sceneNode = node.GetChildObject("sceneobject", flags);

        if (sceneNode)
        {
            if (m_hostInstance.m_params.transform.Deserialise(sceneNode, flags)) { SetDirtyFlags(kDirtyObjectBounds); }
        }
        return m_dirtyFlags;
    }

    __host__ void Host::SceneObject::RecomputeWorldSpaceBoundingBox() 
    {
        m_hostInstance.m_params.worldBBox = m_hostInstance.m_params.objectBBox + m_hostInstance.m_params.transform.trans;
        Log::Warning("Rebuilt world bbox: %s", m_hostInstance.m_params.worldBBox.Format());
    }

    __host__ void Host::SceneObject::RecomputeBoundingBoxes()
    {
        m_hostInstance.m_params.objectBBox = RecomputeObjectSpaceBoundingBox();
        RecomputeWorldSpaceBoundingBox();
    }

}
