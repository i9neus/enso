#include "ProceduralTexture.cuh"

namespace Enso
{
    __host__ __device__ SolidTextureParams::SolidTextureParams() : 
        colour(kOne * 0.5f) {}

    __host__ __device__ SolidTextureParams::SolidTextureParams(const vec3& c) : 
        colour(c) {}
    
    __host__ __device__ GridTextureParams::GridTextureParams() : 
        baseColour(kOne * 0.7f), lineColour(kOne * 0.2f), lineThickness(0.1f), scale(0.1f) {}

    __host__ __device__ GridTextureParams::GridTextureParams(const vec3& base, const vec3& line, const float thickness, const float sca) :
        baseColour(base), lineColour(line), lineThickness(thickness), scale(sca) {}
    
    template<>
    __device__ vec4 Device::ProceduralTexture<SolidTextureParams>::Evaluate(const vec2& uv) const
    {
        return vec4(m_params.colour, 1.f);
    }   

    template<>
    __device__ vec4 Device::ProceduralTexture<GridTextureParams>::Evaluate(const vec2& uv) const
    {
        const vec2 uvScaled = fract(uv / m_params.scale);
        return vec4(mix(m_params.lineColour, m_params.baseColour, step(m_params.lineThickness, cwiseMin(uvScaled))), 1.0f);
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