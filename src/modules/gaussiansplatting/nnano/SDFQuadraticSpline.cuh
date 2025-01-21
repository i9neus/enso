#pragma once

#include "core/2d/bih/BIH.cuh"
#include "core/2d/primitives/QuadraticSpline.cuh"
#include "Beziers.h"

namespace Enso
{    
    namespace Device
    {
        class SDFQuadraticSpline
        {
        public:            
            const BIH2D::BIHData<BIH2D::FullNode>* bih = nullptr;
            Device::Vector<QuadraticSpline>* splineList = nullptr;

        public:
            __host__ __device__ SDFQuadraticSpline() {}

            __host__ __device__ void Validate() const
            {
                CudaAssert(bih);
                CudaAssert(splineList);
            }

            __host__ __device__ void Synchronise(const SDFQuadraticSpline& objects) { *this = objects; }

            __host__ __device__ vec3 Evaluate(const vec2& p) const
            {
                CudaAssertDebug(splineList && bih);

                float distNear = kFltMax;
                vec2 pNear(kFltMax);
                auto onLeaf = [&](const uint* idxRange, const uint* primIdxs) -> float
                {
                    for (int idx = idxRange[0]; idx < idxRange[1]; ++idx)
                    {
                        const vec2 pSpline = (*splineList)[primIdxs[idx]].PerpendicularPoint(p);
                        const float dist = length(pSpline - p);
                        if (dist < distNear)
                        {
                            distNear = dist;
                            pNear = pSpline;
                        }
                    }
                    return distNear;
                };
                BIH2D::TestNearest(*bih, p, onLeaf);

                // Brute-force testing
                /*for (int idx = 0; idx < splineList->size(); ++idx)
                {
                    const vec2 pSpline = (*splineList)[idx].PerpendicularPoint(p);
                    const float dist = length(pSpline - p);
                    if (dist < distNear)
                    {
                        distNear = dist;
                        pNear = pSpline;
                    }
                }*/

                return vec3(distNear, pNear);
            }
             
            __host__ __device__ vec4 DebugBIH(const vec2& pView, const UIViewCtx& viewCtx) const
            {
                // Visualise the spline BVH
                vec4 L(0.);
                const OverlayCtx overlayCtx = OverlayCtx::MakeStroke(viewCtx, vec4(1.f), 3.f);
                auto onLeaf = [&](const uint* idxRange, const uint* primIdxs) -> bool
                {
                    for (int idx = idxRange[0]; idx < idxRange[1]; ++idx)
                    {
                        const auto& spline = (*splineList)[primIdxs[idx]];
                        const vec4 line = spline.EvaluateOverlay(pView, overlayCtx);
                        if (line.w > 0.f) { L = Blend(L, line); }

                        if (Scale(spline.GetBoundingBox(), 0.95).PointOnPerimiter(pView, viewCtx.dPdXY * 10.))
                            L = vec4(kYellow, 1.0f);
                    }

                    return false;
                };
                auto onInner = [&, this](const BBox2f& bBox, const int depth, const bool isLeaf) -> void
                {
                    if (isLeaf && bBox.PointOnPerimiter(pView, viewCtx.dPdXY * 10.))
                        L = vec4(kRed, 1.0f);
                };
                BIH2D::TestPoint(*bih, pView, onLeaf, onInner);

                return L;
            }
        };      
    }

    namespace Host
    {
        class SDFQuadraticSpline : public Host::Asset
        {
        private:
            AssetHandle<Host::BIH2DAsset>                   m_hostBIH;
            AssetHandle<Host::Vector<QuadraticSpline>>      m_hostSplines;

            Device::SDFQuadraticSpline                      m_hostInstance;
            Device::SDFQuadraticSpline                      m_deviceObjects;
            Device::SDFQuadraticSpline*                     cu_deviceInstance;

        public:
            __host__ SDFQuadraticSpline(const Asset::InitCtx& initCtx) :
                Asset(initCtx),
                cu_deviceInstance(AssetAllocator::InstantiateOnDevice<Device::SDFQuadraticSpline>(*this))
            {
                constexpr uint kMinTreePrims = 5;
                m_hostBIH = AssetAllocator::CreateChildAsset<Host::BIH2DAsset>(*this, "bih", kMinTreePrims);
                m_hostSplines = AssetAllocator::CreateChildAsset<Host::Vector<QuadraticSpline>>(*this, "splines");

                // Populate the splines from the embedded data
                auto& splines = *m_hostSplines;
                using namespace FlowBeziers;
                splines.resize(kNumSplines);
                for (int i = 0; i < kNumSplines; ++i)
                {
                    splines[i] = QuadraticSpline(kPoints[i * 2], kPoints[i * 2 + 1]);
                }
                splines.Upload();

                // Create a primitive list ready for building
                // TODO: It's probably faster if we build on the already-sorted index list
                auto& primIdxs = m_hostBIH->GetPrimitiveIndices();
                primIdxs.resize(splines.size());
                for (uint idx = 0; idx < primIdxs.size(); ++idx) { primIdxs[idx] = idx; }

                // Construct the BIH
                std::function<BBox2f(uint)> getPrimitiveBBox = [&splines](const uint& idx) -> BBox2f
                {
                    return Scale(splines[idx].GetBoundingBox(), 1.01f);
                };
                m_hostBIH->Build(getPrimitiveBBox, true);

                // Synchronise the data with the device
                m_deviceObjects.bih = m_hostBIH->GetDeviceData();
                m_deviceObjects.splineList = m_hostSplines->GetDeviceInstance();
                SynchroniseObjects<Device::SDFQuadraticSpline>(cu_deviceInstance, m_deviceObjects);

                // Synchronise the host copy so we can query it
                m_hostInstance.bih = &m_hostBIH->GetHostData();
                m_hostInstance.splineList = &m_hostSplines->GetHostInstance();
            }

            __host__ Host::SDFQuadraticSpline::~SDFQuadraticSpline() noexcept
            {
                AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);

                m_hostBIH.DestroyAsset();
                m_hostSplines.DestroyAsset();
            }

            __host__ inline vec3 Evaluate(const vec2 p) const { return m_hostInstance.Evaluate(p); }

            __host__ Device::SDFQuadraticSpline* GetDeviceInstance() const { return cu_deviceInstance; }
            __host__ Device::SDFQuadraticSpline& GetHostInstance() { return m_hostInstance; }

        };
    }
    
}