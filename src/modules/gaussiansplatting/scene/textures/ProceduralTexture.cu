#include "ProceduralTexture.cuh"

namespace Enso
{
    template<>
    __device__ vec4 Device::ProceduralTexture<SolidTextureParams>::Evaluate(const vec2& uv) const
    {
        return vec4(m_params.colour, 1.f);
    }   
    
    template<typename ParamsType>
    __host__ Host::ProceduralTexture<ParamsType>::ProceduralTexture(const Asset::InitCtx& initCtx, const ParamsType& params) :
        Texture2D(initCtx),
        m_params(params),
        cu_deviceInstance(AssetAllocator::InstantiateOnDevice<Device::ProceduralTexture<ParamsType>>(*this))
    {
        Texture2D::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::Texture2D>(cu_deviceInstance));

        Synchronise(kSyncParams);
    }

    template<typename ParamsType>
    __host__ void Host::ProceduralTexture<ParamsType>::Synchronise(const uint syncFlags)
    {
        if (syncFlags & kSyncParams)
        {
            SynchroniseObjects<Device::ProceduralTexture<ParamsType>>(cu_deviceInstance, m_params);
        }
    }
}