#include "Curve.cuh"

using namespace Cuda;

namespace GI2D
{
    __device__ Device::Curve::Curve()
    {
    }
    
    __device__ void Device::Curve::Synchronise(const Objects& objects)
    {
        m_objects = objects;
    }

    __device__ bool Device::Curve::Intersect(Ray2D& ray, HitCtx2D& hit, float& tFarParent) const
    {
        const RayBasic2D localRay = ToObjectSpace(ray); 
        const Cuda::Device::Vector<LineSegment>& segments = *m_objects.lineSegments;

        auto onIntersect = [&](const uint& startIdx, const uint& endIdx, float& tFarChild) -> float
        {
            for (uint idx = startIdx; idx < endIdx; ++idx)
            {
                if (segments[idx].TestRay(ray, hit))
                {
                    if (hit.tFar < tFarChild && hit.tFar < tFarParent)
                    {
                        tFarChild = hit.tFar;
                    }
                }
            }
        };
        m_objects.bih->TestRay(ray, onIntersect);

        tFarParent = min(tFarParent, hit.tFar);
    }

    __host__ Host::Curve::Curve(const std::string& id) :
        Asset(id),
        cu_deviceInstance(nullptr)
    {
        m_hostBIH = CreateChildAsset<Host::BIH2DAsset>(tfm::format("%s_bih", id), this);
        m_hostLineSegments = CreateChildAsset<Cuda::Host::Vector<LineSegment>>(tfm::format("%s_lineSegments", id), this, kVectorHostAlloc, nullptr);
        
        cu_deviceInstance = InstantiateOnDevice<Device::Curve>(GetAssetID());

        m_deviceData.bih = m_hostBIH->GetDeviceInstance();
        m_deviceData.lineSegments = m_hostLineSegments->GetDeviceInstance();

        Synchronise();
    }

    __host__ Host::Curve::~Curve()
    {
        OnDestroyAsset();
    }

    __host__ void Host::Curve::OnDestroyAsset()
    {
        DestroyOnDevice(GetAssetID(), cu_deviceInstance);

        m_hostBIH.DestroyAsset();
        m_hostLineSegments.DestroyAsset();
    }

    __host__ void Host::Curve::Synchronise()
    {
        SynchroniseObjects(cu_deviceInstance, m_deviceData);
    }
}