#pragma once

#include "../tracables/Tracable.cuh"
#include "../FwdDecl.cuh"

namespace Enso
{
    class LineSegment;

    struct UIInspectorParams
    {
        __host__ __device__ UIInspectorParams() {}
        
        float viewRadius;
    };

    namespace Host { class UIInspector; }

    namespace Device
    {        
        class UIInspector : public Device::Tracable,
                            public UIInspectorParams//,
                            //public AssetTags<Host::UIInspector, Device::UIInspector>
        {
            friend class Host::UIInspector;
        protected:
            __device__ virtual vec4 EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx) const override final;

        public:
            __device__ UIInspector() {}
        };
    }

    namespace Host
    {
        class BIH2DAsset;

        class UIInspector : public Host::Tracable,
                            public UIInspectorParams
        {
        public:
            __host__ UIInspector(const std::string& id);
            __host__ virtual ~UIInspector();

            __host__ virtual void       OnDestroyAsset();

            __host__ virtual uint       OnCreate(const std::string& stateID, const UIViewCtx& viewCtx);
            //__host__ virtual uint       OnSelectElement(const std::string& stateID, const vec2& mousePos, const UIViewCtx& viewCtx, UISelectionCtx& selectCtx);
            __host__ virtual bool       IsConstructed() const;
            __host__ virtual bool       Rebuild(const uint parentFlags, const UIViewCtx& viewCtx);
            __host__ virtual bool       Finalise() override final;

            __host__ static AssetHandle<Host::SceneObject> Instantiate(const std::string& id);
            __host__ static const std::string  GetAssetClassStatic() { return "inspector"; }

            __host__ virtual Device::UIInspector* GetDeviceInstance() const override final
            {
                return cu_deviceInstance;
            }

        private:
            __host__ void Synchronise(const int type);

        private:
            Device::UIInspector*        cu_deviceInstance = nullptr;
            Device::UIInspector         m_hostInstance;

            int                         m_numSelected;
        };
    }
}