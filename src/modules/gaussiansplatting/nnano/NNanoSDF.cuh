#pragma once

#include "core/2d/Ctx.cuh"
#include "core/2d/DrawableObject.cuh"
#include "core/2d/RenderableObject.cuh"

#include "core/assets/DirtinessFlags.cuh"
#include "core/containers/Image.cuh"
#include "core/assets/GenericObject.cuh"
#include "../scene/SceneContainer.cuh"

#include "../FwdDecl.cuh"
//#include "core/3d/BidirectionalTransform.cuh"

namespace Enso
{
    class QuadraticSpline;
    
    struct NNanoSDFParams
    {
        __host__ __device__ NNanoSDFParams();
        __device__ void Validate() const;

        struct
        {
            ivec2 dims;
            BBox2f objectBounds;
        }
        viewport;
    };

    struct NNanoSDFObjects
    {
        __device__ void Validate() const
        {
            CudaAssert(bih);
            CudaAssert(splines);
        }

        BIH2D<BIH2DFullNode>* bih = nullptr;
        Device::Vector<QuadraticSpline>* splines = nullptr;
    };

    namespace Host { class NNanoSDF; }

    namespace Device
    {
        class NNanoSDF : public Device::DrawableObject
        {
            friend Host::NNanoSDF;

        public:
            __host__ __device__ NNanoSDF() {}

            __host__ __device__ void Synchronise(const NNanoSDFParams& params);
            __device__ void Synchronise(const NNanoSDFObjects& objects);

            __host__ __device__ bool            IsClickablePoint(const UIViewCtx& viewCtx) const;
            __host__ __device__ virtual vec4    EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx, const bool isMouseTest) const override final;

        private:


        private:
            NNanoSDFParams            m_params;
            NNanoSDFObjects           m_objects;
        };
    }

    namespace Host
    {
        class NNanoSDF : public Host::DrawableObject
        {
        public:
            __host__                    NNanoSDF(const Asset::InitCtx& initCtx, const AssetHandle<const Host::GenericObjectContainer>& genericObjects);
            __host__ virtual            ~NNanoSDF() noexcept;

            __host__ void               Clear();

            __host__ static AssetHandle<Host::GenericObject> Instantiate(const std::string& id, const Host::Asset& parentAsset, const AssetHandle<const Host::GenericObjectContainer>& genericObjects);
            __host__ static const std::string  GetAssetClassStatic() { return "nnanosdf"; }
            __host__ virtual std::string       GetAssetClass() const override final { return GetAssetClassStatic(); }

            __host__ virtual void       Bind(GenericObjectContainer& objects) override final;
            __host__ virtual bool       Serialise(Json::Node& rootNode, const int flags) const override final;
            __host__ virtual bool       Deserialise(const Json::Node& rootNode, const int flags) override final;
            __host__ virtual BBox2f     ComputeObjectSpaceBoundingBox() override final;
            __host__ virtual bool       HasOverlay() const override { return true; }
            __host__ virtual bool       IsClickablePoint(const UIViewCtx& viewCtx) const override final;
            __host__ virtual bool       IsDelegatable() const override final { return true; }
            __host__ virtual bool       OnDelegateAction(const std::string& stateID, const VirtualKeyMap& keyMap, const UIViewCtx& viewCtx) override final;

            __host__ virtual Device::NNanoSDF* GetDeviceInstance() const override final
            {
                return cu_deviceInstance;
            }

        protected:
            __host__ virtual void       OnSynchroniseDrawableObject(const uint syncFlags) override final;
            __host__ virtual bool       OnCreateDrawableObject(const std::string& stateID, const UIViewCtx& viewCtx, const vec2& mousePosObject) override final;
            __host__ virtual bool       OnRebuildDrawableObject() override final;

        private:
            __host__ void               CreateScene();

            template<typename T>
            __host__ void           KernelParamsFromImage(const AssetHandle<Host::Image<T>>& image, dim3& blockSize, dim3& gridSize) const
            {
                const auto& meta = image->GetMetadata();
                blockSize = dim3(16, 16, 1);
                gridSize = dim3((meta.Width() + 15) / 16, (meta.Height() + 15) / 16, 1);
            }

            Device::NNanoSDF* cu_deviceInstance = nullptr;
            Device::NNanoSDF                m_hostInstance;
            NNanoSDFObjects                 m_deviceObjects;
            NNanoSDFParams                  m_params;

            AssetHandle<Host::BIH2DAsset>                   m_hostBIH;
            AssetHandle<Host::Vector<QuadraticSpline>>      m_hostSplines;

            HighResolutionTimer               m_renderTimer;
            HighResolutionTimer               m_redrawTimer;

        };
    }
}