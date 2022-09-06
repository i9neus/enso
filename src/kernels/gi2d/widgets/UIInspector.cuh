#pragma once

#include "../SceneObject.cuh"

#define FWD_DECL_VECTOR
#define FWD_DECL_BIH2D
#include "../FwdDecl.cuh"

using namespace Cuda;

namespace GI2D
{
    class LineSegment;

    struct UIInspectorParams
    {
        SceneObjectParams sceneObject;

        float viewRadius;
    };

    namespace Host { class UIInspector; }

    namespace Device
    {
        class UIInspector : public Device::SceneObject,
                            public Cuda::AssetTags<Host::UIInspector, Device::UIInspector>
        {
        public:
            struct Objects
            {
                Device::BIH2DAsset* bih = nullptr;
                Cuda::Device::Vector<LineSegment>* lineSegments = nullptr;
            };

            __device__ virtual bool EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx, vec4& L) const override final;

        public:
            __device__ UIInspector() {}

            __device__ void             Synchronise(const Objects& objects);
            __device__ void             Synchronise(const UIInspectorParams& params);

        private:
            UIInspectorParams           m_params;

            float                       m_radius;
        };
    }

    namespace Host
    {
        class BIH2DAsset;

        class UIInspector : public Host::SceneObject,
                            public Cuda::AssetTags<Host::UIInspector, Device::UIInspector>
        {
        public:
            __host__ UIInspector(const std::string& id);
            __host__ virtual ~UIInspector();

            __host__ virtual void       OnDestroyAsset() override final;
            __host__ void               SynchroniseParams();

            __host__ virtual uint       OnCreate(const std::string& stateID, const UIViewCtx& viewCtx) override final;
            //__host__ virtual uint       OnSelectElement(const std::string& stateID, const vec2& mousePos, const UIViewCtx& viewCtx, UISelectionCtx& selectCtx) override final;
            __host__ virtual bool       IsConstructed() const override final;
            __host__ virtual bool       Rebuild(const uint parentFlags, const UIViewCtx& viewCtx) override final;
            __host__ virtual bool       Finalise() override final;

            __host__ static AssetHandle<GI2D::Host::SceneObject> Instantiate(const std::string& id);
            __host__ static const std::string  GetAssetTypeString() { return "inspector"; }

        private:


        private:
            Device::UIInspector*                            cu_deviceInstance;
            Device::UIInspector::Objects                    m_deviceData;

            int                                             m_numSelected;
            UIInspectorParams                               m_params;

        };
    }
}