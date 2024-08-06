#pragma once

#include "NLM.cuh"

namespace Enso
{
    __device__ NLMDenoiser::NLMDenoiser() :
        m_meanAccumBuffer(nullptr),
        m_varAccumBuffer(nullptr),
        m_viewportDims(ivec2(0)),
        m_kN(5),
        m_kM(0),
        m_kAlpha(0.5f),
        m_kK(0.5f)
    {
    }

    __host__ __device__ void NLMDenoiser::Initialise(const int N, const int M, const float alpha, const float K)
    {
        m_kN = N;
        m_kM = M;
        m_kAlpha = alpha;
        m_kK = K;
    }

    __device__ void NLMDenoiser::Initialise(Device::ImageRGBW* meanAccumBuffer, Device::ImageRGBW* varAccumBuffer)
    {
        CudaAssert(meanAccumBuffer);
        CudaAssert(varAccumBuffer);

        m_meanAccumBuffer = meanAccumBuffer;
        m_varAccumBuffer = varAccumBuffer;
        m_viewportDims = ivec2(meanAccumBuffer->Dimensions());
    }

    __device__ __forceinline__ bool NLMDenoiser::IsValidTexel(const ivec2& p)
    {
        return p.x >= 0 && p.x < m_viewportDims.x && p.y >= 0 && p.y < m_viewportDims.y;
    }

    __device__ __forceinline__ bool NLMDenoiser::GetTexel(const ivec2& p, vec3& P, vec3& varP)
    {
        if (!IsValidTexel(p)) { return false; }
        else
        {
            const auto& T = m_meanAccumBuffer->At(p);
            P = T.xyz / fmaxf(1.f, T.w);
            const auto& varT = m_varAccumBuffer->At(p);
            varP = varT.xyz / sqr(fmaxf(1.f, varT.w));

            return true;
        }
    }

    __device__ __forceinline__ float NLMDenoiser::PatchDistance(const ivec2& p, const ivec2& q)
    {
        float sumWeights = 0.;
        int sumTexels = 0;
        for (int v = -m_kM; v <= m_kM; ++v)
        {
            for (int u = -m_kM; u <= m_kM; ++u)
            {
                const ivec2 pp = p + ivec2(u, v);
                const ivec2 qq = q + ivec2(u, v);

                vec3 P, varP, Q, varQ;
                if (GetTexel(pp, P, varP) && GetTexel(qq, Q, varQ))
                {
                    // Compute the distance from N to M
                    const vec3 d2 = (sqr(P - Q) - sqr(m_kAlpha) * (varP + min(varP, varQ))) / (sqr(m_kK) * (vec3(1e-15f) + varP + varQ));

                    // Derive the weight from the distance
                    sumWeights += 1.0 / (1e-10f + expf(-fmaxf(0.0f, cwiseMax(d2))));
                    sumTexels++;
                }  
            }
        }

        //return sumWeights / float(sqr(kM*2+1));
        return float(sumTexels) / sumWeights;
    }

    __device__ vec3 NLMDenoiser::FilterPixel(const ivec2& p)
    {
        CudaAssert(m_meanAccumBuffer);
        CudaAssert(m_varAccumBuffer);
        
        float sumWeights = 0.;
        vec3 sumL = kZero;
        for (int v = -m_kN; v <= m_kN; ++v)
        {
            for (int u = -m_kN; u <= m_kN; ++u)
            {
                const ivec2 ij = p + ivec2(u, v);
                if (IsValidTexel(ij))
                {
                    const vec4& T = m_meanAccumBuffer->At(ij);

                    const float weight = (u == 0 && v == 0) ? 1.f : PatchDistance(p, ij);
                    sumL += weight * T.xyz / fmaxf(1.f, T.w);;
                    sumWeights += weight;
                }
            }
        }

        return sumL / sumWeights;
    }

    __device__ vec3 NLMDenoiser::FilterPixelBox(const ivec2& p)
    {        
        CudaAssert(m_meanAccumBuffer);
        CudaAssert(m_varAccumBuffer);
        
        vec3 sumL = kZero;
        int sumWeights = 0;
        for (int v = -m_kN; v <= m_kN; ++v)
        {
            for (int u = -m_kN; u <= m_kN; ++u)
            {
                const ivec2 ij = p + ivec2(u, v);
                if (IsValidTexel(ij))
                {
                    const auto& T = m_meanAccumBuffer->At(ij);
                    sumL += T.xyz / fmaxf(1.f, T.w);
                    sumWeights++;
                }
            }
        }

        return sumL / float(sumWeights);
    }
}