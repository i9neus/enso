#pragma once

#include "Texture2D.cuh"

namespace Enso
{
    struct SolidTextureParams
    {
        __host__ __device__ SolidTextureParams();
        __host__ __device__ SolidTextureParams(const vec3& c);
        __device__ void Validate() const {}

        vec3 colour;
    };

    struct GridTextureParams
    {
        __host__ __device__ GridTextureParams();
        __host__ __device__ GridTextureParams(const vec3& base, const vec3& line, const float thickness, const float sca);
        __device__ void Validate() const {}

        vec3 baseColour;
        vec3 lineColour;
        float lineThickness;
        float scale;
    };
    
    namespace Device
    {
        // This class provides an interface for querying the tracable via geometric operations
        template<typename ParamsType>
        class ProceduralTexture : public Device::Texture2D
        {
        public:
            __device__ ProceduralTexture() {}
            __device__ virtual vec4                 Evaluate(const vec2& uv) const override final;
            __device__ void                         Synchronise(const ParamsType& params) { m_params = params; }

        private:
            ParamsType                              m_params;
        };
    }

    namespace Host
    {
        template<typename ParamsType>
        class ProceduralTexture : public Host::Texture2D
        {
        public:
            __host__                            ProceduralTexture(const Asset::InitCtx& initCtx, const ParamsType& params);

            __host__ void                       Synchronise(const uint syncFlags);
            __host__ Device::ProceduralTexture<ParamsType>* GetDeviceInstance() { return cu_deviceInstance; }

        protected:
            Device::ProceduralTexture<ParamsType>* cu_deviceInstance = nullptr;

        private:
            ParamsType                          m_params;
        };

        template class ProceduralTexture<SolidTextureParams>;

        using SolidTexture = ProceduralTexture<SolidTextureParams>;
    }
}