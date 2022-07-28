#include "CudaTestCard.cuh"
#include "kernels/math/CudaColourUtils.cuh"

namespace Cuda
{
    TestCard::TestCard(const std::string& id) :
        Asset(id),
        m_wheel(0.0)
    {
    }

    __global__ void KernelComposite(Device::ImageRGBA* deviceOutputImage, float wheel)
    {
        const ivec2 p = kKernelPos<ivec2>();
        if (p.x >= deviceOutputImage->Width()|| p.y >= deviceOutputImage->Height()) { return; }
        
        const float tone = (((p.x / 10) + (p.y / 10)) % 2 == 0) ? 0.5 : 0.7;
        deviceOutputImage->At(p) = vec4(Hue(wheel) * tone, 1.0f);
    }

    __host__ void TestCard::Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const
    {
        const auto& meta = hostOutputImage->GetMetadata();
        dim3 blockSize(16, 16, 1);
        dim3 gridSize((meta.Width() + 15) / 16, (meta.Height() + 15) / 16, 1);

        KernelComposite << < gridSize, blockSize, 0, m_hostStream >> > (hostOutputImage->GetDeviceInstance(), m_wheel * 0.01);
    }

    __host__ void TestCard::OnDestroyAsset()
    {

    }
}