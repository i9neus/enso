#pragma once

#include "Texture2D.cuh"

namespace Enso
{
    struct TextureMapParams
    {
        __device__ void Validate() const;        

        cudaTextureObject_t texture = 0;
        size_t width = 0u;
        size_t height = 0u;
    };
    
    namespace Device
    {
        // This class provides an interface for querying the tracable via geometric operations
        class TextureMap : public Device::Texture2D
        {
        public:
            __device__ TextureMap() {}
            __device__ virtual vec4                 Evaluate(const vec2& uv) const override final;
            __device__ void                         Synchronise(const TextureMapParams& params) { m_params = params; }

        protected:
            __device__ vec4                         Lerp(const vec2& xy) const;

        private:
            TextureMapParams                        m_params;
            
        };
    }

    namespace Host
    {
        class TextureMap : public Host::Texture2D
        {
        public:
            __host__                            TextureMap(const Asset::InitCtx& initCtx);
            __host__                            TextureMap(const Asset::InitCtx& initCtx, const std::string& path);
            __host__ virtual                    ~TextureMap();

            __host__ Device::TextureMap*        GetDeviceInstance() { return cu_deviceInstance; }
            __host__ void                       Load(const std::string& path);
            __host__ void                       Unload();

        protected:
            Device::TextureMap*                 cu_deviceInstance = nullptr;

        private:
            TextureMapParams                    m_textureParams;
            cudaArray_t                         cu_textureArray;

        };
    }
}