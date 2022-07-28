#include "../CudaAsset.cuh"
#include "../CudaManagedObject.cuh"
#include "../CudaImage.cuh"

namespace Cuda
{
    class TestCard : public Host::Asset
    {
    public:
        TestCard(const std::string& id);

        __host__ void Composite(AssetHandle<Host::ImageRGBA>& hostOutputImage) const;
        __host__ void OnDestroyAsset();

        float m_wheel;
    };
}