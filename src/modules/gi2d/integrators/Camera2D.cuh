#pragma once

#include "core/Asset.cuh"
#include "../Transform2D.cuh"
#include "../RenderCtx.cuh"
#include "../SceneObject.cuh"

namespace Enso
{
    namespace Device
    {
        class Camera2D : public Device::SceneObject
        {
        public:
            __host__ __device__ Camera2D() {}

            __device__ virtual bool CreateRay(Ray2D& ray, HitCtx2D& hit, RenderCtx& renderCtx) const = 0;

            __device__ virtual void Accumulate(const vec4& L, const RenderCtx& ctx) = 0;
        };
    }

    namespace Host
    {
        class Camera2D : public Host::SceneObject
        {
        public:
            __host__ virtual bool       Rebuild(const uint parentFlags, const UIViewCtx& viewCtx) = 0;
            __host__ virtual Device::Camera2D* GetDeviceInstance() const = 0;

            __host__ virtual bool       Serialise(Json::Node& rootNode, const int flags) const override;
            __host__ virtual uint       Deserialise(const Json::Node& rootNode, const int flags) override;

        protected:
            __host__ Camera2D(const std::string& id, Device::Camera2D& hostInstance) :
                SceneObject(id, hostInstance),
                m_hostInstance(hostInstance)
            {
            }

            template<typename SubType> __host__ inline void Synchronise(SubType* deviceInstance, const int syncFlags)
            {
                SceneObject::Synchronise(deviceInstance, syncFlags);
            }

        protected:
            struct
            {
                vec2                        dragAnchor;
                bool                        isDragging;
            }
            m_onMove;

            Device::Camera2D& m_hostInstance;
        };
    }
}      