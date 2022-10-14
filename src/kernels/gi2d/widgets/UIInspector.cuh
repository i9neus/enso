#pragma once

#include "../tracables/Tracable.cuh"

#define FWD_DECL_VECTOR
#define FWD_DECL_BIH2D
#include "../FwdDecl.cuh"

using namespace Cuda;

namespace GI2D
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
        class UIInspector : public TracableInterface,
                            public UIInspectorParams,
                            public Cuda::AssetTags<Host::UIInspector, Device::UIInspector>
        {
        protected:
            __device__ virtual bool EvaluatePrimitives(const vec2& pWorld, const UIViewCtx& viewCtx, vec4& L) const;

        public:
            __device__ UIInspector() {}
        };
    }

    namespace Host
    {
        class BIH2DAsset;

        class UIInspector : public Host::Tracable,
                            public UIInspectorParams,
                            public Cuda::AssetTags<Host::UIInspector, Device::UIInspector>
        {
        public:
            __host__ UIInspector(const std::string& id);
            __host__ virtual ~UIInspector();

            __host__ virtual void       OnDestroyAsset() override final;

            __host__ virtual uint       OnCreate(const std::string& stateID, const UIViewCtx& viewCtx) override final;
            //__host__ virtual uint       OnSelectElement(const std::string& stateID, const vec2& mousePos, const UIViewCtx& viewCtx, UISelectionCtx& selectCtx) override final;
            __host__ virtual bool       IsConstructed() const override final;
            __host__ virtual bool       Rebuild(const uint parentFlags, const UIViewCtx& viewCtx) override final;
            __host__ virtual bool       Finalise() override final;

            __host__ static AssetHandle<GI2D::Host::SceneObject> Instantiate(const std::string& id);
            __host__ static const std::string  GetAssetTypeString() { return "inspector"; }

        private:
            __host__ void Synchronise(const int type);

        private:
            Device::UIInspector*                            cu_deviceInstance;

            int                                             m_numSelected;
        };
    }
}