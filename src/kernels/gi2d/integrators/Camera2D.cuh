#pragma once

#include "kernels/CudaAsset.cuh"
#include "../Transform2D.cuh"
#include "../RenderCtx.cuh"

namespace GI2D
{
    namespace Device
    {
        class Camera2D : public Cuda::Device::Asset
        {
        public:
            __host__ __device__ Camera2D() {}

            __device__ virtual bool CreateRay(Ray2D& ray, RenderCtx& renderCtx) const = 0;
        };

        class OrthographicCamera2D : public Camera2D
        {
        public:
            __host__ __device__ OrthographicCamera2D() {}

            __device__ void Prepare(const ViewTransform2D& viewTransform, const BBox2f& sceneBounds, const int downsample)
            {
                m_viewTransform = viewTransform;
                m_sceneBounds = sceneBounds;
                m_downsample = downsample;
            }
            
            __device__ virtual bool CreateRay(Ray2D& ray, RenderCtx& renderCtx) const override final
            {
                // Transform from screen space to view space
                ray.o = m_viewTransform.matrix * vec2(kKernelPos<ivec2>() * m_downsample);
                if (!m_sceneBounds.Contains(ray.o)) { return false; }
                
                // Randomly scatter
                const float theta = renderCtx.rng.Rand<0>() * kTwoPi;
                ray.d = vec2(cosf(theta), sinf(theta));

                return true;
            }

        private:
            ViewTransform2D     m_viewTransform;
            BBox2f              m_sceneBounds;
            int                 m_downsample;
        };
    }


}