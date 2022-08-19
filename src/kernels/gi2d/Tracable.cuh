#pragma once

#include "CudaPrimitive2D.cuh"
#include "BIH2DAsset.cuh"
#include "Transform2D.cuh"
#include "UICtx.cuh"

#include "../CudaRenderObject.cuh"

using namespace Cuda;

namespace GI2D
{
    namespace Device
    {
        class Tracable : public Cuda::Device::RenderObject
        {
        public:
            __device__ virtual bool Intersect(Ray2D& ray, HitCtx2D& hit, float& tFar) const = 0;

        protected:
            __device__ Tracable() {};

            __device__ __forceinline__ RayBasic2D ToObjectSpace(const Ray2D& world) const
            {
                RayBasic2D obj;
                obj.o = world.o - m_transform.trans;
                obj.d = world.d + obj.o;
                obj.o = m_transform.fwd * obj.o;
                obj.d = (m_transform.fwd * obj.d) - obj.o;
                return obj;
            }

        protected:
            BidirectionalTransform2D m_transform;
        };
    }

    namespace Host
    {
        class Tracable : public Cuda::Host::RenderObject
        {
        public:
            __host__ virtual uint       OnCreate(const std::string& stateID, const vec2& mousePos) = 0;
            __host__ virtual uint       OnSelect() = 0;
            __host__ virtual uint       OnDeselect() = 0;
            __host__ virtual uint       OnSelectElement(const std::string& stateID, const vec2& mousePos, const UIViewCtx& viewCtx, UISelectionCtx& selectCtx) = 0;

        protected:
            __host__ Tracable(const std::string& id) : RenderObject(id) {}

            BidirectionalTransform2D m_transform;
        };
    }
}