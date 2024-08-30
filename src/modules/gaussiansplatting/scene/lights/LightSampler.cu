#include "LightSampler.cuh"

namespace Enso
{    
    __host__  Host::LightSampler::LightSampler(const InitCtx& initCtx) : 
        SceneObject(initCtx), 
        cu_deviceInstance(nullptr) 
    {
        Listen({ kDirtyParams });
    }
    
    __host__ void Host::LightSampler::OnDirty(const DirtinessEvent& flag, AssetHandle<Host::Asset>& caller)
    {
        // If the signaller is the tracable, update the parameters of the sampler transform and signal a resync
        AssetHandle<Host::Tracable> tracable(m_weakTracable);
        if(tracable == caller)
        {
            m_params.transform = tracable->GetTransform();
            SignalDirty(kDirtyParams);
        }
    }

    __host__ bool Host::LightSampler::BindTracable(AssetHandle<Host::Tracable>& tracable)
    {
        if (!TryBind(tracable))
        {
            Log::Error("Error: cannot bind sampler '%s' to primitive '%s'; primitive is the wrong type for the sampler.");
            return false;
        }
        else
        {
            // Flag the tracable as a light and retrieve its transform
            tracable->MakeLight(m_params.radiance);
            m_params.transform = tracable->GetTransform();
            Synchronise(kSyncParams);

            // We need to listen out for changes made to the tracable so we can update the sampler.
            m_weakTracable = tracable.GetWeakHandle();
        }
    }

    __host__ void Host::LightSampler::Synchronise(const uint syncFlags)
    {
        if (syncFlags & kSyncParams)
        {
            SynchroniseObjects<Device::LightSampler>(cu_deviceInstance, m_params);
        }
    }
}