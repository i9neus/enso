#include "CudaJitterableParameter.cuh"
#include "generic/JsonUtils.h"

#include <random>

namespace Cuda
{    
    template<typename PType>
    __host__ void JitterableScalar<PType>::FromJson(const std::string& id, const ::Json::Node& node, const uint flags)
    {
        std::vector<float> data;
        if (!node.GetArrayValues(id, data, flags)) { return; }

        if (data.size() == 0)
        {
            Json::ReportError(flags, "Warning: jitterable scalar '%s' should have at least 1 element.\n", id);
            return;
        }
        
        p = data[0];
        dpdt = (data.size() >= 2) ? data[1] : 0.0f;
        t = (data.size() >= 3) ? data[2] : 0.0f;

        Evaluate();
    }

    template<typename PType>
    __host__ void JitterableScalar<PType>::Evaluate()
    { 
        // For integral types, increase the sampleable value by one so we capture the full range
        eval =(std::is_integral<PType>::value) ? 
                PType(p + (dpdt + 0.99999f) * (t * 2.0f - 1.0f)) : 
                PType(p + dpdt * (t * 2.0f - 1.0f));
    }

    template<typename PType>
    __host__ void JitterableScalar<PType>::ToJson(const std::string& id, ::Json::Node& node) const
    {
        node.AddArray(id, std::vector<float>({ float(p), float(dpdt), float(t) }));
    }

    template<typename PType>
    __host__ void JitterableScalar<PType>::Randomise(vec2 range)
    {        
        // Clamp and constrain 
        range[1] = clamp(range[1], 1.0f, 1.0f);
        range[0] = clamp(range[0], 0.0f, range[1]);

        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<> rng(range[0], range[1]);

        t = rng(mt);

        Evaluate();
    }

    template<typename PType, typename TType>
    __host__ void JitterableVec<PType, TType>::FromJson(const std::string& id, const ::Json::Node& node, const uint flags)
    {
        std::vector<std::vector<float>> matrix;
        if (!node.GetArray2DValues(id, matrix, flags)) { return; }

        // Check the integrity of the data
        if (matrix.size() == 0)
        {
            Json::ReportError(flags, "Error: jitterable vec%i '%s': matrix must have at least one row.\n", PType::kDims, id);
            return;
        }
        for (int row = 0; row < matrix.size(); ++row)
        {
            if (matrix[row].size() != PType::kDims)
            {
                Json::ReportError(flags, "Error: jitterable vec%i '%s': row %i should have %i columns but found %i.\n", PType::kDims, id, PType::kDims, matrix[row].size());
                return;
            }
        }

        // Copy the data
        for (int col = 0; col < PType::kDims; ++col)
        {
            p[col] = matrix[0][col];

            dpdt[col] = (matrix.size() >= 2) ? matrix[1][col] : 0.0f;
            t[col] = (matrix.size() == 3) ? matrix[2][col] : 0.5f;
        }

        Evaluate();
    }

    template<typename PType, typename TType>
    __host__ void JitterableVec<PType, TType>::ToJson(const std::string& id, ::Json::Node& node) const
    {
        std::vector<std::vector<float>> matrix(3, std::vector<float>(PType::kDims));
        for (int col = 0; col < PType::kDims; ++col)
        {
            matrix[0][col] = p[col];
            matrix[1][col] = dpdt[col];
            matrix[2][col] = t[col];
        }

        node.AddArray2D(id, matrix);
    }

    template<typename PType, typename TType>
    __host__ void JitterableVec<PType, TType>::Randomise(vec2 range)
    {
        // Clamp and constrain 
        range[1] = clamp(range[1], range[0], 1.0f);
        range[0] = clamp(range[0], 0.0f, range[1]);

        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<> rng(range[0], range[1]);

        for (int i = 0; i < TType::kDims; ++i) { t[i] = rng(mt); }
        
        Evaluate();
    }

    template<typename PType, typename TType>
    __host__ void JitterableVec<PType, TType>::Evaluate()
    {
        // For integral types, increase the sampleable value by one so we capture the full range
        eval = (std::is_integral<typename PType::kType>::value) ?
                PType(p + (dpdt + 0.99999f) * (t * 2.0f - 1.0f)) :
                PType(p + dpdt * (t * 2.0f - 1.0f));
    }

    __host__ void JitterableFlags::FromJson(const std::string& id, const ::Json::Node& node, const uint flags)
    {
        std::vector<std::vector<int>> matrix;
        if (!node.GetArray2DValues(id, matrix, flags)) { return; }

        if (matrix.size() == 0)
        {
            Json::ReportError(flags, "Warning: jitterable flags '%s' should have at least 1 element.\n", id);
            return;
        }
        
        p = dpdt = t = 0;
        for (int bit = 0; bit < validBits; ++bit)
        {
            if (bit < matrix[0].size() && matrix[0][bit] != 0) { p |= 1 << bit; }
            if (matrix.size() >= 2 && bit < matrix[1].size() && matrix[1][bit] != 0) { dpdt |= 1 << bit; }
            if (matrix.size() >= 3 && bit < matrix[2].size() && matrix[2][bit] != 0) { t |= 1 << bit; }
        }

        Evaluate();
    }

    __host__ void JitterableFlags::ToJson(const std::string& id, ::Json::Node& node) const
    {
        std::vector<std::vector<int>> matrix(3, std::vector<int>(validBits));
        for (int bit = 0; bit < validBits; ++bit)
        {
            matrix[0][bit] = (p >> bit) & 1;
            matrix[1][bit] = (dpdt >> bit) & 1;
            matrix[2][bit] = (t >> bit) & 1;
        }

        node.AddArray2D(id, matrix);
    }

    __host__ void JitterableFlags::Evaluate()
    {
        eval = (p & ~dpdt) | (dpdt & t);
    }

    __host__ void JitterableFlags::Randomise(vec2 range)
    {
        // Clamp and constrain 
        range[1] = clamp(range[1], 1.0f, 1.0f);
        range[0] = clamp(range[0], 0.0f, range[1]);

        std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_real_distribution<> rng(range[0], range[1]);

        t = 0;
        for (int bit = 0; bit < validBits; ++bit)
        {
            t |= uint(rng(mt) > 0.5f) << bit;
        }

        Evaluate();
    }
}