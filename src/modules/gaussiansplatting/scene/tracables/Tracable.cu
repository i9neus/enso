#include "Tracable.cuh"
#include "../lights/LightSampler.cuh"

namespace Enso
{
    __host__ Host::Tracable::Tracable(const InitCtx& initCtx, const BidirectionalTransform& transform, const int materialIdx) :
        Host::SceneObject(initCtx)
    {
        m_params.transform = transform;
        m_params.materialIdx = materialIdx;
    }

    __host__ void Host::Tracable::MakeLight(const vec3& radiance)
    {
        m_params.isLight = true;
        m_params.radiance = radiance;
        Synchronise(kSyncParams);    
    }
}