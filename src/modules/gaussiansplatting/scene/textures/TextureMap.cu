#include "TextureMap.cuh"

#include "core/containers/Vector.cuh"
#include "io/FilesystemUtils.h"
#include "io/images/Exr.h"

namespace Enso
{
    __device__ void TextureMapParams::Validate() const
    {
        
    }

    __device__ vec4 Device::TextureMap::Evaluate(const vec2& uv) const
    {
        return vec4(tex2D<float4>(m_params.texture, uv.x, uv.y));
    }   
    
    __host__ Host::TextureMap::TextureMap(const Asset::InitCtx& initCtx) :
        Texture2D(initCtx),
        cu_textureArray(nullptr),
        cu_deviceInstance(AssetAllocator::InstantiateOnDevice<Device::TextureMap>(*this))
    {
        Texture2D::SetDeviceInstance(AssetAllocator::StaticCastOnDevice<Device::Texture2D>(cu_deviceInstance));
    }

    __host__ Host::TextureMap::TextureMap(const Asset::InitCtx& initCtx, const std::string& path) :
        TextureMap(initCtx)
    {
        Load(path);
    }

    __host__ Host::TextureMap::~TextureMap()
    {
        Unload();
    }

    __host__ void Host::TextureMap::Load(const std::string& path)
    {
        Unload();
        
        // Load the data from the EXR file     
        Enso::ImageIO::Exr exrData(path);
        AssertMsgFmt(exrData && exrData.width > 0 && exrData.height > 0, "Failed to load texture '%s'", path);

        AssetAllocator::GuardedAllocDevice2DArray<float4>(*this, exrData.width, exrData.height, cu_textureArray);

        const float* pixelData = *exrData;
        const size_t pitch = exrData.width * sizeof(float) * 4;

        // Copy data located at address h_data in host memory to device memory
        IsOk(cudaMemcpy2DToArray(cu_textureArray, 0, 0, pixelData, pitch, exrData.width * sizeof(float) * 4, exrData.height, cudaMemcpyHostToDevice));

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cu_textureArray;

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 1;

        // Create texture object
        IsOk(cudaCreateTextureObject(&m_textureParams.texture, &resDesc, &texDesc, NULL));

        // Synchronise everything with the device
        m_textureParams.width = exrData.width;
        m_textureParams.height = exrData.height;
        SynchroniseObjects<Device::TextureMap>(cu_deviceInstance, m_textureParams); 
    }

    __host__ void Host::TextureMap::Unload()
    {
        AssetAllocator::GuardedFreeDeviceTextureObject(m_textureParams.texture);
        AssetAllocator::GuardedFreeDevice2DArray<float4>(*this, m_textureParams.width, m_textureParams.height, cu_textureArray); 

        m_textureParams.width = 0;
        m_textureParams.height = 0;
    }
}