#pragma once

#include "Common.cuh"

#include "Transform2D.cuh"
#include "UICtx.cuh"
#include "FwdDecl.cuh"

#include "kernels/CudaRenderObject.cuh"
#include "kernels/CudaVector.cuh"
#include "BIH2DAsset.cuh"

namespace GI2D
{

    using namespace Cuda;

    template<typename ObjectType, typename ParamsType>
    __global__ static void KernelSynchroniseObjects3(ObjectType* cu_object, const size_t hostParamsSize, const ParamsType* cu_params)
    {
        // Check that the size of the object in the device matches that of the host. Empty base optimisation can bite us here. 
        assert(cu_object);
        assert(cu_params);
        assert(sizeof(ParamsType) == hostParamsSize);

        ParamsType& cast = static_cast<ParamsType&>(*cu_object);
        cast = *cu_params;
    }

    template<typename ParamsType, typename ObjectType>
    __host__ void SynchroniseObjects3(ObjectType* cu_object, const ParamsType& params)
    {
        //AssertIsTransferrableType<ParamsType>();
        Assert(cu_object);

        ParamsType* cu_params;
        IsOk(cudaMalloc(&cu_params, sizeof(ParamsType)));
        IsOk(cudaMemcpy(cu_params, &params, sizeof(ParamsType), cudaMemcpyHostToDevice));

        IsOk(cudaDeviceSynchronize());
        KernelSynchroniseObjects3 << <1, 1 >> > (cu_object, sizeof(ParamsType), cu_params);
        IsOk(cudaDeviceSynchronize());

        IsOk(cudaFree(cu_params));
    }

    enum AssetSyncType : int { kSyncObjects = 1, kSyncParams = 2 };

    enum SceneObject2Flags : uint
    {
        kSceneObject2Selected = 1u
    };

    struct SceneObjectParams2
    {
        __host__ __device__ SceneObjectParams2() {}

        BBox2f                      m_objectBBox;
        BBox2f                      m_worldBBox;

        BidirectionalTransform2D    m_transform;
        uint                        m_attrFlags;
    };

    // This class provides an interface for querying the tracable via geometric operations
    class SceneObjectInterface2 : public SceneObjectParams2
    {
    public:
        __device__ virtual bool                             EvaluateOverlay(const vec2& p, const UIViewCtx& viewCtx, vec4& L) const { return false; }

        __host__ __device__ const BBox2f& GetObjectSpaceBoundingBox() const { return m_objectBBox; };
        __host__ __device__ const BBox2f& GetWorldSpaceBoundingBox() const { return m_worldBBox; };

    protected:
        __host__ __device__ SceneObjectInterface2() {}

        __device__ bool                                     EvaluateControlHandles(const vec2& pWorld, const UIViewCtx& viewCtx, vec4& L) const;

        __host__ __device__ __forceinline__ RayBasic2D ToObjectSpace(const Ray2D& world) const
        {
            return m_transform.RayToObjectSpace(world);
        }

    private:
        BBox2f m_handleInnerBBox;
    };

    namespace Device
    {
        class SceneObject2 : virtual public SceneObjectInterface2,
            public Cuda::Device::RenderObject
        {
        public:
            __device__ SceneObject2() {}
        };
    }

    namespace Host
    {
        class SceneObject2 : virtual public SceneObjectInterface2,
            public Cuda::Host::RenderObject
        {
        public:
            __host__ virtual uint       OnCreate(const std::string& stateID, const UIViewCtx& viewCtx) = 0;
            //__host__ virtual uint       OnSelectElement(const std::string& stateID, const UIViewCtx& viewCtx, UISelectionCtx& selectCtx) = 0;
            __host__ virtual uint       OnMove(const std::string& stateID, const UIViewCtx& viewCtx);
            __host__ virtual uint       OnSelect(const bool isSelected);

            __host__ virtual bool       Finalise() = 0;

            __host__ virtual bool       IsConstructed() const = 0;
            __host__ virtual bool       Rebuild(const uint parentFlags, const UIViewCtx& viewCtx) = 0;

            __host__ uint               GetDirtyFlags() const { return m_dirtyFlags; }
            __host__ bool               IsFinalised() const { return m_isFinalised; }
            __host__ bool               IsSelected() const { return m_attrFlags & kSceneObject2Selected; }

            __host__ SceneObjectInterface2* GetDeviceInstance() const
            {
                AssertMsgFmt(cu_deviceSceneObjectInterface2, "SceneObjectInterface2::cu_deviceSceneObjectInterface2 has not been initialised by its inheriting class '%s'", GetAssetID().c_str());
                return cu_deviceSceneObjectInterface2;
            }

            __host__ virtual void SetAttributeFlags(const uint flags, bool isSet = true)
            {
                if (SetGenericFlags(m_attrFlags, flags, isSet))
                {
                    SetDirtyFlags(kGI2DDirtyUI, true);
                }
            }

        protected:
            __host__ SceneObject2(const std::string& id);

            template<typename SubType>
            __host__ void Synchronise(SubType* cu_object, const int type)
            {
                if (type == kSyncParams) { SynchroniseObjects3<SceneObjectParams2>(cu_object, *this); }
            }

            __host__ void SetDirtyFlags(const uint flags, const bool isSet = true) { SetGenericFlags(m_dirtyFlags, flags, isSet); }
            __host__ void ClearDirtyFlags() { m_dirtyFlags = 0; }

        protected:
            uint                        m_dirtyFlags;
            bool                        m_isFinalised;

            struct
            {
                vec2                        dragAnchor;
                bool                        isDragging;
            }
            m_onMove;

            SceneObjectInterface2* cu_deviceSceneObjectInterface2 = nullptr;
        };
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////


    // This class provides an interface for querying the tracable via geometric operations
    class TracableInterface2 : virtual public SceneObjectInterface2
    {
    public:
        __host__ __device__ virtual bool                    IntersectRay(Ray2D& ray, HitCtx2D& hit) const { return false; }
        //__host__ __device__ virtual bool                    InteresectPoint(const vec2& p, const float& thickness) const { return false; }
        __host__ __device__ virtual bool                    IntersectBBox(const BBox2f& bBox) const;

        //__host__ __device__ virtual vec2                    PerpendicularPoint(const vec2& p) const { return vec2(0.0f); }

        __device__ virtual bool                             EvaluateOverlay(const vec2& pWorld, const UIViewCtx& viewCtx, vec4& L) const override final { return EvaluatePrimitives(pWorld, viewCtx, L); }
        __host__ __device__ virtual const vec3              GetColour() const { return kOne; }

    protected:
        __host__ __device__ TracableInterface2() {}

        __device__ virtual bool                             EvaluatePrimitives(const vec2& pWorld, const UIViewCtx& viewCtx, vec4& L) const { return false; }

        __host__ __device__ __forceinline__ RayBasic2D ToObjectSpace(const Ray2D& world) const
        {
            return m_transform.RayToObjectSpace(world);
        }

    private:
        BBox2f m_handleInnerBBox;
    };

    namespace Host
    {
        class Tracable2 : public TracableInterface2,
            public Host::SceneObject2,
            public Cuda::AssetTags<Host::Tracable2, TracableInterface2>
        {
        public:
            __host__ TracableInterface2* GetDeviceInstance() const { return cu_deviceTracableInterface2; }

        protected:
            __host__ Tracable2(const std::string& id) : SceneObject2(id) {}

            template<typename SubType> __host__ void Synchronise(SubType* deviceData, const int syncType) { SceneObject2::Synchronise(deviceData, syncType); }

        protected:
            TracableInterface2* cu_deviceTracableInterface2 = nullptr;

            struct
            {
                vec2                        dragAnchor;
                bool                        isDragging;
            }
            m_onMove;
        };
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    class LineSegment;

    struct CurveObjects2
    {
        BIH2D<BIH2DFullNode>* m_bih = nullptr;
        Core::Vector<LineSegment>* m_lineSegments = nullptr;
    };

    namespace Device
    {
        class Curve2 : public TracableInterface2,
            public CurveObjects2
        {
        public:
            __device__ Curve2() {}

            __host__ __device__ virtual bool    IntersectRay(Ray2D& ray, HitCtx2D& hit) const override final;

        protected:
            __device__ virtual bool             EvaluatePrimitives(const vec2& pWorld, const UIViewCtx& viewCtx, vec4& L) const override final;
        };
    }

    namespace Host
    {
        class BIH2DAsset;

        class Curve2 : public Host::Tracable2,
            public CurveObjects2,
            public Cuda::AssetTags<Host::Curve2, Device::Curve2>
        {
        public:
            __host__ Curve2(const std::string& id);
            __host__ virtual ~Curve2();

            __host__ virtual void       OnDestroyAsset() override final;
            __host__ void               Synchronise(const int syncType);

            __host__ virtual uint       OnCreate(const std::string& stateID, const UIViewCtx& viewCtx) override final;
            //__host__ virtual uint       OnSelectElement(const std::string& stateID, const vec2& mousePos, const UIViewCtx& viewCtx, UISelectionCtx& selectCtx) override final;
            __host__ virtual bool       IsConstructed() const override final;
            __host__ virtual bool       Rebuild(const uint parentFlags, const UIViewCtx& viewCtx) override final;
            __host__ virtual bool       Finalise() override final;

            __host__ Device::Curve2* GetDeviceInstance() const
            {
                return cu_deviceInstance;
            }

            __host__ static void Test();

        private:


        private:
            Device::Curve2* cu_deviceInstance = nullptr;
            CurveObjects2                                    m_deviceObjects;

            AssetHandle<GI2D::Host::BIH2DAsset>                   m_hostBIH;
            AssetHandle<Cuda::Host::Vector<LineSegment>>    m_hostLineSegments;

            int                                             m_numSelected;

        };
    }









    using TracableContainer2 = Cuda::Host::AssetVector<Host::Tracable2, TracableInterface2>;
    using InspectorContainer2 = Cuda::Host::AssetVector<Host::Tracable2, TracableInterface2>;


    struct UILayerParams2
    {
        __host__ __device__ UILayerParams2()
        {
            m_selectionCtx.isLassoing = false;
        }

        UIViewCtx           m_viewCtx;
        UISelectionCtx      m_selectionCtx;
    };

    namespace Host
    {
        class UILayer2 : public Cuda::Host::AssetAllocator,
            public UILayerParams2
        {
        public:
            UILayer2(const std::string& id, AssetHandle<GI2D::Host::BIH2DAsset>& bih, AssetHandle<TracableContainer2>& tracables) :
                AssetAllocator(id),
                m_hostBIH(bih),
                m_hostTracables(tracables)
            {
            }

            virtual ~UILayer2() = default;

            __host__ virtual void   Render() = 0;
            __host__ virtual void   Composite(AssetHandle<Cuda::Host::ImageRGBA>& hostOutputImage) const = 0;

            __host__ void Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx)
            {
                m_viewCtx = viewCtx;
                m_selectionCtx = selectionCtx;
                m_dirtyFlags = dirtyFlags;
            }

            __host__ void           SetDirtyFlags(const uint flags) { m_dirtyFlags |= flags; }

        protected:
            template<typename SubType>
            __host__ void Synchronise(SubType* cu_object, const int syncType)
            {
                if (syncType & kSyncParams) { SynchroniseObjects3<UILayerParams2>(cu_object, *this); }
            }

            template<typename T>
            __host__ void           KernelParamsFromImage(const AssetHandle<Cuda::Host::Image<T>>& image, dim3& blockSize, dim3& gridSize) const
            {
                const auto& meta = image->GetMetadata();
                blockSize = dim3(16, 16, 1);
                gridSize = dim3((meta.Width() + 15) / 16, (meta.Height() + 15) / 16, 1);
            }

        protected:
            AssetHandle<GI2D::Host::BIH2DAsset>                     m_hostBIH;
            AssetHandle<TracableContainer2>                         m_hostTracables;

            uint                                                    m_dirtyFlags;
        };
    }

    struct OverlayParams2
    {
        __host__ __device__ OverlayParams2();

        UIGridCtx           m_gridCtx;
    };

    struct OverlayObjects2
    {
        __host__ __device__ OverlayObjects2() {}
        
        Generic::Vector<TracableInterface2*>* m_tracables = nullptr;
        Generic::Vector<TracableInterface2*>* m_inspectors = nullptr;
        BIH2D<BIH2DFullNode>* m_bih = nullptr;
        Cuda::Device::ImageRGBW* m_accumBuffer = nullptr;
    };

    namespace Device
    {
        class Overlay2 : public Cuda::Device::Asset,
            public UILayerParams2,
            public OverlayParams2,
            public OverlayObjects2
        {
        public:
            __host__ __device__ Overlay2();

            __device__ void Render();
            __device__ void Composite(Cuda::Device::ImageRGBA* outputImage);
        };
    }

    namespace Host
    {
        class Overlay2 : public UILayer2,
            public OverlayParams2
        {
        public:
            Overlay2(const std::string& id, AssetHandle<Host::BIH2DAsset>& bih, AssetHandle<TracableContainer2>& tracables, AssetHandle<InspectorContainer2>& inspectors,
                const uint width, const uint height, cudaStream_t renderStream);

            virtual ~Overlay2();

            __host__ virtual void Render() override final;
            __host__ virtual void Composite(AssetHandle<Cuda::Host::ImageRGBA>& hostOutputImage) const override final;
            __host__ void Rebuild(const uint dirtyFlags, const UIViewCtx& viewCtx, const UISelectionCtx& selectionCtx);

            __host__ void OnDestroyAsset();

        protected:
            __host__ void Synchronise(const int syncType);

        private:
            //__host__ void TraceRay();

            Device::Overlay2* cu_deviceData = nullptr;
            OverlayObjects2                              m_deviceObjects;

            AssetHandle<Cuda::Host::ImageRGBW>          m_hostAccumBuffer;
            AssetHandle<InspectorContainer2>            m_hostInspectors;
        };
    }

    void InvokeDebugger();
}