#include "../math/CudaMath.cuh"

namespace Cuda
{
    __host__ __device__ __forceinline__ float ErfApprox(const float x)
    {
        // Closed-form approxiation of the error function.
        // See 'Uniform Approximations for Transcendental Functions', Winitzki 2003, https://doi.org/10.1007/3-540-44839-X_82

        constexpr float a = 8 * (kPi - 3) / (3 * kPi * (4 - kPi));
        return copysign(1.0f, x) * sqrt(1 - expf(-(x * x) * (4 / kPi + a * x * x) / (1 + a * x * x)));
    }

    __host__ __device__ __forceinline__ float AntiderivGauss(const float x, const float sigma)
    {
        // The antiderivative of the normalised Gaussian
        return 0.5f * (1.0f + ErfApprox(x / (sigma * kRoot2)));
    }

    __host__ __device__ float Integrate1DGaussian(const float t0, const float t1, const float radius)
    {
        // Integrates a Gaussian function whose standard deviation is preset such that its integral between [-radius, radius] is almost 1
        constexpr float sigma = 0.35f;
        return (AntiderivGauss(t1 / radius, sigma) - AntiderivGauss(t0 / radius, sigma));
    }

    __host__ std::vector<float> GenerateGaussianKernel1D(const int kernelSpan)
    {
        // Generates a normalised Gaussian kernel of arbitrary size. The normal distribution is unbounded so we choose a value
        // of sigma that, when integrated, captures >99% of the area under the curve in the range x = [-1, 1].

        Assert(kernelSpan % 2 == 1);

        std::vector<float> kernel;
        kernel.resize(kernelSpan);

        const int kernelRadius = kernelSpan / 2;
        constexpr float sigma = 0.35f;
        constexpr float recipIntegralNorm = 1 / 0.995841f;

        for (int i = 0; i <= kernelRadius; ++i)
        {
            // Compute the sub-interval [x0, x1] that corresponds to the bucket of the kernel element
            const float x0 = (i / float(kernelSpan)) * 2.0 - 1.0f;
            const float x1 = ((i + 1) / float(kernelSpan)) * 2.0 - 1.0f;

            // Integrate the Gaussian
            kernel[i] = kernel[kernelSpan - i - 1] = (AntiderivGauss(x1, sigma) - AntiderivGauss(x0, sigma)) * recipIntegralNorm;
        }

        return kernel;
    }

    __host__ std::vector<float> GenerateBoxKernel1D(const int kernelSpan)
    {
        Assert(kernelSpan % 2 == 1);

        return std::vector<float>(kernelSpan, 1.0f / kernelSpan);
    }
}