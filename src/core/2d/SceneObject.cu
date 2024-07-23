#include "core/2d/SceneObject.cuh"
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

    // FIXME: hostInstance causes segfault when passed by reference. Find out why.
    __host__ Host::SceneObject::SceneObject(const Asset::InitCtx& initCtx, Device::SceneObject* hostInstance, const AssetHandle<const Host::SceneContainer>& scene) :
        GenericObject(initCtx),
        m_hostInstance(*hostInstance)
    {

    }

    __host__ void Host::SceneObject::SetDeviceInstance(Device::SceneObject* deviceInstance)
    {
        cu_deviceInstance = deviceInstance;
    }

    __host__ void Host::SceneObject::SetTransform(const vec2& trans)
    {
        m_hostInstance.m_params.transform.trans = trans;
    }

    __host__ bool Host::SceneObject::OnCreate(const std::string& stateID, const UIViewCtx& viewCtx)
    {
        // Handle the open event by initialising the transform
        if (stateID == "kCreateSceneObjectOpen")
        {
            m_hostInstance.m_params.transform.trans = viewCtx.mousePos;
            SignalDirty({ kDirtyObjectRebuild });
        }

        // Call the virtual method implemented by inheriting classes. 
        if (OnCreateSceneObject(stateID, viewCtx, viewCtx.mousePos - m_hostInstance.m_params.transform.trans))
        {
            // Recompute the bounding boxes and signal the need to rebuild
            RecomputeBoundingBoxes();            
            SignalDirty({ kDirtyObjectRebuild });
        }

        return true;
    }

    __host__ void Host::SceneObject::RecomputeBoundingBoxes()
    {
        m_hostInstance.m_params.objectBBox = ComputeObjectSpaceBoundingBox();
        m_hostInstance.m_params.worldBBox = m_hostInstance.m_params.objectBBox + m_hostInstance.m_params.transform.trans;
        
        SignalDirty(kDirtyObjectBoundingBox);
    }
    
    __host__ bool Host::SceneObject::OnMove(const std::string& stateID, const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx)
    {
        if (stateID == "kMoveSceneObjectBegin")
        {
            m_onMove.dragAnchorDelta = m_hostInstance.m_params.transform.trans - viewCtx.mousePos;
            Log::Error("kMoveSceneObjectBegin");
        }
        else if (stateID == "kMoveSceneObjectDragging")
        {
            m_hostInstance.m_params.transform.trans = viewCtx.mousePos + m_onMove.dragAnchorDelta;
            Log::Warning("kMoveSceneObjectDragging");            
        }
        else if (stateID == "kMoveSceneObjectEnd")
        {
            Log::Success("kMoveSceneObjectEnd");
        }
        else
        {
            return false;
        }

        RecomputeBoundingBoxes(); 
        return true;
    }

    __host__ void Host::SceneObject::Synchronise(const uint syncFlags)
    {
        if (syncFlags & kSyncParams)
        {
            SynchroniseObjects<Device::SceneObject>(cu_deviceInstance, m_hostInstance.m_params);
        }

        OnSynchroniseSceneObject(syncFlags);
    }

    __host__ bool Host::SceneObject::Serialise(Json::Node& node, const int flags) const
    {
        Json::Node sceneNode = node.AddChildObject("sceneobject");

        m_hostInstance.m_params.transform.Serialise(sceneNode, flags);
        return true;
    }

    __host__ bool Host::SceneObject::Deserialise(const Json::Node& node, const int flags)
    {
        const Json::Node sceneNode = node.GetChildObject("sceneobject", flags);

        bool isDirty = false;
        if (sceneNode)
        {
            isDirty |= m_hostInstance.m_params.transform.Deserialise(sceneNode, flags);
        }

        return isDirty;
    }

    __host__ bool Host::SceneObject::OnSelect(const bool isSelected)
    {
        SetGenericFlags(m_hostInstance.m_params.attrFlags, uint(kSceneObjectSelected), isSelected);
        SignalDirty(kDirtyParams);
        return true;
    }

    __host__ bool Host::SceneObject::Rebuild()
    {
        // If inheriting objects that we're doing a rebuild
        if (OnRebuildSceneObject())
        {
            // Update the transform and re-sync everything
            RecomputeBoundingBoxes();
            Synchronise(kSyncParams | kSyncObjects);

            return true;
        
        }
        return false;
    }
}
