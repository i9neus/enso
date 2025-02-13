#pragma once

#include "PDF2.cuh"

namespace Enso
{   
    __global__ void KernelPrepareCDF(Device::DualImage3f* pdf)
    {
        CudaAssertDebug(pdf);
        if (kKernelX < pdf->Width() - 1 && kKernelY < pdf->Height())
        {
            float* element = pdf->At(kKernelX, kKernelY);

            // This index will contain the normalised PDF
            element[0] = fmaxf(0.f, element[0]);

            // This index will contain the normalised CMF
            // Initialise the values of the CMF at the point corresponding its accumulated interval. x = 0 is the left-most point so it always has a cumulative mass of 0.
            element[1] = (kKernelX == 0) ? 0.f : (0.5f * (fmaxf(0.f, pdf->At(kKernelX, kKernelY)[0]) + fmaxf(0.f, pdf->At(kKernelX - 1, kKernelY)[0])));
        }
    }

    __global__ void KernelIntegrateRows(Device::DualImage3f* pdf, const int iterIdx)
    {
        CudaAssertDebug(pdf);
        if (kKernelX < pdf->Width() - 1 && kKernelY < pdf->Height())
        {
            // Iterative reduce. Needs O(log n) texture taps instead of O(n)        
            if (((kKernelX >> iterIdx) & 1) == 1)
            {
                const uint subCDFPrev = (uint(kKernelX) & ~((1u << iterIdx) - 1u)) - 1u;
                if (subCDFPrev != 0xffffffffu)
                {
                    pdf->At(kKernelX, kKernelY)[1] += pdf->At(subCDFPrev, kKernelY)[1];
                }
            }
        }
    }

    __global__ void KernelIntegrateCol(Device::DualImage3f* pdf, vec2* norm, const int iterIdx)
    {
        CudaAssertDebug(pdf && norm);
        const float height = pdf->Height();
        if (kKernelIdx < height)
        {
            const int width = pdf->Width();
            // On the first pass through the data, initialise the CMF values similar to KernelPrepareCDF()
            if (iterIdx == 0)
            {
                float* f = pdf->At(width - 1, kKernelIdx);
                
                // Cache the original sum of the element of this row so we can normalise them in a later pass
                f[2] = pdf->At(width - 2, kKernelIdx)[1];

                // Divide the accumulated value of the row by its length to maintain precision.
                f[0] = f[2] / (width - 1);
                
                // Pre-integrate the CMF
                f[1] = (kKernelIdx == 0) ? 0.f : (0.5f * (f[0] + pdf->At(width - 2, kKernelIdx - 1)[1] / (width - 1)));  
            }

            if (((kKernelIdx >> iterIdx) & 1) == 1)
            {
                const uint subCDFPrev = (uint(kKernelIdx) & ~((1u << iterIdx) - 1u)) - 1u;
                if (subCDFPrev != 0xffffffffu)
                {
                    pdf->At(width - 1, kKernelIdx)[1] += pdf->At(width - 1, subCDFPrev)[1];
                }
            }

            // Cache the accumuated value of the CMF so we can use it to normalise each entry in the column
            if (kKernelIdx == height - 1)
            {
                const float sum = fmaxf(1e-10f, pdf->At(pdf->Width() - 1, kKernelIdx)[1]);
                *norm = vec2(sum / height, sum);
            }
        }
    }
     
    __global__ void KernelNormaliseCol(Device::DualImage3f* pdf, const vec2* norm)
    {
        CudaAssertDebug(pdf && norm);
        if (kKernelIdx < pdf->Height())
        {
            // Normalise the column PDF and CMF by the norm
            float* f = pdf->At(pdf->Width() - 1, kKernelIdx);
            f[0] /= norm->y;
            f[1] /= norm->y;
        }
    }

    __global__ void KernelNormaliseRows(Device::DualImage3f* pdf, const vec2* norm)
    {
        CudaAssertDebug(pdf && norm);
        if (kKernelX < pdf->Width() - 1 && kKernelY < pdf->Height())
        {
            float* f = pdf->At(kKernelX, kKernelY);
            float* fLast = pdf->At(pdf->Width() - 1, kKernelY);

            // Normalise the CDF by the maximum accumulated value in the row
            f[1] /= fmaxf(1e-10f, fLast[2]);

            // Normalise the PDF by the maximum accumulate row value and the weight of the entire row relative to the entire PDF
            f[0] /= fmaxf(1e-10f, norm->x);
        }
    }
     
    __device__ void Device::PDF2::Set(const int x, const int y, const float pdf, const float value)
    {
        CudaAssertDebug(x >= 0 && x < m_width&& y >= 0 && y < m_height);
        m_cdf->As<vec3>(x, y) = vec3(pdf, 0, value);
    }

    __host__ __device__ __forceinline__ float PiecewiseLinearSample(float y0, float y1, float xi)
    {
        constexpr float kEpsilon = 1e-6f;
        const float m = y1 - y0;
        if (fabsf(m) < kEpsilon)
        { 
            return xi; 
        }
        else
        {
            xi *= 0.5f * (y0 + y1);
            float r = ((-y0 + sqrt(y0 * y0 + 2.f * m * xi)) / m);
            /*if (isnan(r))
            {
                float i = y0 * y0 + 2.f * m * xi;
                printf("%.20f, %.20f, %.20f -> %.20f\n", y0, y1, xi, i);
            }*/
            return r;
        }
    }

    __device__ vec3 Device::PDF2::Sample(const vec2& xi, vec2& p) const
    {
        // Lower bound theta    
        int y0 = 0, y1 = m_height - 1;
        while (y1 - y0 > 1)
        {
            int h = y0 + (y1 - y0) / 2;
            float fh = m_cdf->At(m_width, h)[1];
            if (xi.y <= fh) { y1 = h; }
            else { y0 = h; }
        }

        const vec3 fy0 = m_cdf->As<vec3>(m_width, y0);
        const vec3 fy1 = m_cdf->As<vec3>(m_width, y1);

        float dy = PiecewiseLinearSample(fy0.x, fy1.x, (xi.y - fy0.y) / (fy1.y - fy0.y));
        //float dy = (xi.y - fy0.y) / (fby.y - fy0.y);   // Piecewise constant

        // Lower bound
        int x0 = 0, x1 = m_width - 1;
        while (x1 - x0 > 1)
        {
            int h = x0 + (x1 - x0) / 2;
            float fh = mix(m_cdf->At(h, y0)[1], m_cdf->At(h, y1)[1], dy);
            if (xi.x <= fh) { x1 = h; }
            else { x0 = h; }
        }

        const vec3 t00 = m_cdf->As<vec3>(x0, y0);
        const vec3 t10 = m_cdf->As<vec3>(x1, y0);
        const vec3 t01 = m_cdf->As<vec3>(x0, y1);
        const vec3 t11 = m_cdf->As<vec3>(x1, y1);

        const vec3 fx0 = mix(t00, t01, dy);
        const vec3 fx1 = mix(t10, t11, dy); 

        const float dx = PiecewiseLinearSample(fx0.x, fx1.x, (xi.x - fx0.y) / (fx1.y - fx0.y));

        //float dx = (xi.x - fx0.y) / (fx1.y - fx0.y);  // Piecewise constant

        p = vec2((float(x0) + dx) / float(m_width - 1),
                 (float(y0) + dy) / float(m_height - 1));

        // Return the PDF
        return mix(mix(t00, t10, dx), mix(t01, t11, dx), dy);
    }

    __device__ vec3 Device::PDF2::Evaluate(vec2 p) const
    {
        ivec2 i;
        vec2 d;
        for (int c = 0; c < 2; ++c)
        {
            if (p[c] <= 0.) { i[c] = 0; d[c] = 0.; }
            else if (p[c] >= 1.) { i[c] = m_dims[c] - 2; d[c] = 1.; }
            else
            {
                p[c] *= m_dims[c] - 1;
                i[c] = int(p[c]);
                d[c] = fract(p[c]);
            }
        }

        return mix(mix(m_cdf->As<vec3>(i.x, i.y), m_cdf->As<vec3>(i.x + 1, i.y), d.x),
                   mix(m_cdf->As<vec3>(i.x, i.y + 1), m_cdf->As<vec3>(i.x + 1, i.y + 1), d.x),
                   d.y);
    }

    __host__ Host::PDF2::PDF2(const Asset::InitCtx& initCtx, const int width, const int height, cudaStream_t stream) :
        Asset(initCtx),
        cu_deviceInstance(AssetAllocator::InstantiateOnDevice<Device::PDF2>(*this)),
        m_cudaStream(stream)
    {
        AssertMsgFmt(width > 0 && height > 0 && (width & (width - 1)) == 0 && (height & (height - 1)) == 0, "Dimensions %i x %i must be powers of two.", width, height);

        m_cdf = AssetAllocator::CreateChildAsset<Host::DualImage3f>(*this, "cdf", width + 1, height);
        m_cdf->Erase();

        SynchroniseTrivialParams(cu_deviceInstance, m_cdf->GetDeviceInstance());
    }

    __host__ Host::PDF2::~PDF2()
    {
        AssetAllocator::DestroyOnDevice(*this, cu_deviceInstance);
    }

    __host__ void Host::PDF2::Rebuild()
    {
        const int numRowReductions = std::log2(1 + m_cdf->Width());
        const int numColReductions = std::log2(1 + m_cdf->Height());        

        // Prepare by copying the PDF values at [0] into the CDF values at [1] so they can integrated
        auto [gridDims2D, blockDims2D] = Get2DLaunchParams(m_cdf->Width(), m_cdf->Height());    
        KernelPrepareCDF << <gridDims2D, blockDims2D, 0, m_cudaStream >> > (m_cdf->GetDeviceInstance());

        // Reduce the row CDFs
        for (int i = 0; i < numRowReductions; ++i)
        {
            KernelIntegrateRows << <gridDims2D, blockDims2D, 0, m_cudaStream >> > (m_cdf->GetDeviceInstance(), i);
        }

        // Reduce the column CDF
        auto [gridDims1D, blockDims1D] = Get1DLaunchParams(m_cdf->Height());
        for (int i = 0; i < numColReductions; ++i)
        {
            KernelIntegrateCol << <gridDims1D, blockDims1D, 0, m_cudaStream >> > (m_cdf->GetDeviceInstance(), m_colNorm.GetDeviceData(), i);
        }

        // Normalise the column first so we know the relative magnitude of each row 
        KernelNormaliseCol << <gridDims1D, blockDims1D, 0, m_cudaStream >> > (m_cdf->GetDeviceInstance(), m_colNorm.GetDeviceData());
        // Then normalise the rows so we can compute the absolute PDF per entry
        KernelNormaliseRows << <gridDims2D, blockDims2D, 0, m_cudaStream >> > (m_cdf->GetDeviceInstance(), m_colNorm.GetDeviceData());

        auto& norm = m_colNorm.Download();
        Log::Error("PDF norm: %f, %f", norm.x, norm.y);

        IsOk(cudaStreamSynchronize(m_cudaStream));     
    }

}