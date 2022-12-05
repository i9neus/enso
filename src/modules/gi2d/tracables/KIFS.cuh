#pragma once

#include "Tracable.cuh"
#include "../FwdDecl.cuh"

namespace Enso
{
    struct KIFSParams
    {
        __host__ __device__ KIFSParams();

        struct
        {
            float       rotate;
            vec2        pivot;
            float       isosurface;
            float       sdfScale;
            float       iterScale;
            int         numIterations;
             
            float       objectBounds;
        }
        m_kifs;
      
    };

    namespace Device
    {
        class KIFS : public Device::Tracable,
                     public KIFSParams
        {
        public:
            __host__ __device__ KIFS();

            __host__ __device__ virtual bool    IntersectRay(const Ray2D& ray, HitCtx2D& hit) const override final;
            __device__ virtual vec4             EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx) const override final;
            __device__ virtual void             OnSynchronise(const int) override final;

            __host__ __device__ __forceinline__ bool  Iterate(vec2 z, vec3& F) const;

        private:
            const mat2 m_kBary;
            const mat2 m_kBaryInv;
            
            mat2       m_rot1;
            mat2       m_rot2;
        };
    }

    namespace Host
    {
        class BIH2DAsset;

        class KIFS : public Host::Tracable
        {
        public:
            __host__ KIFS(const std::string& id);
            __host__ virtual ~KIFS();

            __host__ virtual void       OnDestroyAsset() override final;
            __host__ void               Synchronise(const int syncType);

            __host__ virtual uint       OnCreate(const std::string& stateID, const UIViewCtx& viewCtx) override final;

            //__host__ virtual uint       OnSelectElement(const std::string& stateID, const vec2& mousePos, const UIViewCtx& viewCtx, UISelectionCtx& selectCtx) override final;
            __host__ virtual bool       Rebuild(const uint parentFlags, const UIViewCtx& viewCtx) override final;

            __host__ virtual Device::KIFS* GetDeviceInstance() const override final
            {
                return cu_deviceInstance;
            }

            __host__ static AssetHandle<Host::GenericObject> Instantiate(const std::string& id, const Json::Node&);
            __host__ static const std::string GetAssetClassStatic() { return "kifs"; }
            __host__ virtual std::string GetAssetClass() const override final { return GetAssetClassStatic(); }

            __host__ virtual bool       Serialise(Json::Node& rootNode, const int flags) const override final;
            __host__ virtual uint       Deserialise(const Json::Node& rootNode, const int flags) override final;

        protected:
            __host__ bool               Finalise();
            __host__ virtual BBox2f     RecomputeObjectSpaceBoundingBox() override final;

        private:
            Device::KIFS*               cu_deviceInstance = nullptr;
            Device::KIFS                m_hostInstance;       
        };
    }

    // Explicitly declare instances of this class for its inherited types
    //template class Host::Tracable<Device::KIFS>;
    //template class Host::GenericObject<Device::KIFS>;
}