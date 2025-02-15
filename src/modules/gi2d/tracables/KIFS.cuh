#pragma once

#include "Tracable.cuh"
#include "../FwdDecl.cuh"

namespace Enso
{
    struct KIFSDebugData
    {
        enum __attrs : int { kMaxPoints = 10 };
        vec2    pNear, pFar;
        vec2    marchPts[kMaxPoints];
        vec2    normal;
        vec2    hit;
        bool    isHit;
    };
    
    struct KIFSParams
    {
        __host__ __device__ KIFSParams();
        __device__ void Validate() const {}

        struct
        {
            float       rotate;
            vec2        pivot;
            float       isosurface;
            float       sdfScale;
            float       iterScale;
            int         numIterations;             
            float       objectBounds;
            float       primSize;

        } kifs;
      
        struct
        {
            float       cutoffThreshold;
            float       escapeThreshold;
            float       failThreshold;
            float       rayIncrement;
            float       rayKickoff;
            int         maxIterations;

        } intersector;

        struct
        {
            float       phase;
            float       range;

        } look;
    };

    namespace Host { class KIFS; }

    namespace Device
    {         
        class KIFS : public Device::Tracable
        {
            friend class Host::KIFS;

        public:
            __host__ __device__ KIFS();

            __host__ __device__ virtual bool    IntersectRay(const Ray2D& ray, HitCtx2D& hit) const override final;
            __host__ __device__ uint            OnMouseClick(const UIViewCtx& viewCtx) const;

            __host__ __device__ virtual vec4    EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx, const bool isMouseTest) const override final;
            
            __host__ __device__ void            Synchronise(const KIFSParams& params);

            __host__ __device__ __forceinline__ vec3 EvaluateSDF(vec2 z, const mat2& basis, uint& code) const;

        private:
            KIFSParams m_params;

            const mat2 m_kBary;
            const mat2 m_kBaryInv;
            
            mat2       m_rot1;
            mat2       m_rot2;
        };
    }

    namespace Host
    {
        class BIH2DAsset;

        class KIFS final : public Host::Tracable
        {
        public:
            __host__ KIFS(const Asset::InitCtx& initCtx);
            __host__ virtual ~KIFS() noexcept;

            __host__ virtual void       OnSynchroniseTracable(const uint syncType) override final;
            __host__ virtual uint       OnMouseClick(const UIViewCtx& viewCtx) const override final;
            __host__ virtual bool       OnRebuildSceneObject() override final { return true; }

            __host__ virtual Device::KIFS* GetDeviceInstance() const override final
            {
                return cu_deviceInstance;
            }

            __host__ static AssetHandle<Host::GenericObject> Instantiate(const std::string& id, const Host::Asset& parentAsset, const AssetHandle<const Host::SceneContainer>& scenes);
            __host__ static const std::string GetAssetClassStatic() { return "kifs"; }
            __host__ virtual std::string GetAssetClass() const override final { return GetAssetClassStatic(); }

            __host__ virtual bool       Serialise(Json::Node& rootNode, const int flags) const override final;
            __host__ virtual bool       Deserialise(const Json::Node& rootNode, const int flags) override final;

            __host__ virtual BBox2f     ComputeObjectSpaceBoundingBox() override final;

        protected:
            __host__ virtual bool       OnCreateSceneObject(const std::string& stateID, const UIViewCtx& viewCtx, const vec2& mousePosObject) override final;

        private:
            Device::KIFS*               cu_deviceInstance = nullptr;
            Device::KIFS                m_hostInstance;       
        };
    }

    // Explicitly declare instances of this class for its inherited types
    //template class Host::Tracable<Device::KIFS>;
    //template class Host::GenericObject<Device::KIFS>;
}