#include "core/2d/DrawableObject.cuh"
#include "core/math/ColourUtils.cuh"
#include "io/json/JsonUtils.h"

namespace Enso
{
    __device__ bool Device::DrawableObject::EvaluateControlHandles(const vec2& pWorld, const UIViewCtx& viewCtx, vec4& L) const
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
    __host__ Host::DrawableObject::DrawableObject(const Asset::InitCtx& initCtx, Device::DrawableObject* hostInstance) :
        GenericObject(initCtx),
        m_hostInstance(*hostInstance)
    {

    }

    __host__ void Host::DrawableObject::SetDeviceInstance(Device::DrawableObject* deviceInstance)
    {
        cu_deviceInstance = deviceInstance;
    }

    __host__ void Host::DrawableObject::SetTransform(const vec2& trans)
    {
        m_hostInstance.m_params.transform.trans = trans;
    }

    __host__ void Host::DrawableObject::Verify() const
    {
        // This method should be called after construction to verify that all derived classes have initialised the appropriate objects
        AssertMsg(cu_deviceInstance, "cu_deviceInstance is nullptr. Did you forget to call DrawableObject::SetDeviceInstance() in from the derived class constructor?");
    }

    __host__ bool Host::DrawableObject::OnCreate(const std::string& stateID, const UIViewCtx& viewCtx)
    {
        // Handle the open event by initialising the transform
        if (stateID == "kCreateDrawableObjectOpen")
        {
            m_hostInstance.m_params.transform.trans = viewCtx.mousePos;
        }

        // Call the virtual method implemented by inheriting classes. 
        if (OnCreateDrawableObject(stateID, viewCtx, viewCtx.mousePos - m_hostInstance.m_params.transform.trans))
        {
            // Recompute the bounding boxes and signal the need to rebuild
            RecomputeBoundingBoxes();  
        }

        return true;
    }

    __host__ void Host::DrawableObject::RecomputeBoundingBoxes()
    {
        m_hostInstance.m_params.objectBBox = ComputeObjectSpaceBoundingBox();
        m_hostInstance.m_params.worldBBox = m_hostInstance.m_params.objectBBox + m_hostInstance.m_params.transform.trans;
        
        SignalDirty(kDirtyViewportObjectBBox);
    }
    
    __host__ bool Host::DrawableObject::OnMove(const std::string& stateID, const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx)
    {
        if (stateID == "kMoveDrawableObjectBegin")
        {
            m_onMove.dragAnchorDelta = m_hostInstance.m_params.transform.trans - viewCtx.mousePos;
            Log::Error("kMoveDrawableObjectBegin");
        }
        else if (stateID == "kMoveDrawableObjectDragging")
        {
            m_hostInstance.m_params.transform.trans = viewCtx.mousePos + m_onMove.dragAnchorDelta;
            Log::Warning("kMoveDrawableObjectDragging");            
        }
        else if (stateID == "kMoveDrawableObjectEnd")
        {
            Log::Success("kMoveDrawableObjectEnd");
        }
        else
        {
            return false;
        }

        RecomputeBoundingBoxes(); 
        return true;
    }

    __host__ void Host::DrawableObject::Synchronise(const uint syncFlags)
    {
        if (syncFlags & kSyncParams)
        {
            AssertMsg(cu_deviceInstance, "cu_deviceInstance is nullptr. Did you forget to call DrawableObject::SetDeviceInstance() in from the derived class constructor?");
            SynchroniseObjects<Device::DrawableObject>(cu_deviceInstance, m_hostInstance.m_params);
        }

        OnSynchroniseDrawableObject(syncFlags);
    }

    __host__ bool Host::DrawableObject::Serialise(Json::Node& node, const int flags) const
    {
        Json::Node sceneNode = node.AddChildObject("sceneobject");

        m_hostInstance.m_params.transform.Serialise(sceneNode, flags);
        return true;
    }

    __host__ bool Host::DrawableObject::Deserialise(const Json::Node& node, const int flags)
    {
        const Json::Node sceneNode = node.GetChildObject("sceneobject", flags);

        bool isDirty = false;
        if (sceneNode)
        {
            isDirty |= m_hostInstance.m_params.transform.Deserialise(sceneNode, flags);
        }

        return isDirty;
    }

    __host__ bool Host::DrawableObject::OnSelect(const bool isSelected)
    {
        SetGenericFlags(m_hostInstance.m_params.attrFlags, uint(kDrawableObjectSelected), isSelected);
        SignalDirty(kDirtyParams);
        return true;
    }

    __host__ bool Host::DrawableObject::Rebuild()
    {
        // If inheriting objects that we're doing a rebuild
        if (!OnRebuildDrawableObject()) { return false; }      

        // Update the transform and re-sync everything
        RecomputeBoundingBoxes();
        Synchronise(kSyncParams | kSyncObjects);

        return true;
    }

    __host__ uint Host::DrawableObject::OnMouseClick(const VirtualKeyMap& keyMap, const UIViewCtx& viewCtx) const 
    { 
        if (IsClickablePoint(viewCtx))
        {
            // If the object is already selected and the drawable supports delegation, signal a shift into delegation mode
            // Otherwise, insta-select it and shift into dragging mode
            if (IsSelected() && IsDelegatable()) { return kDrawableObjectDelegatedAction; }
            else { return kDrawableObjectPrecisionDrag; }
        }
        else
        {
            return kDrawableObjectInvalidSelect;
        }
        
        return kDrawableObjectInvalidSelect; 
    }
}
