#pragma once

#include "../CudaAsset.cuh"

namespace Json
{
    class Node;
    class Document;
}

namespace Core
{
    class Serialisable
    {
    public: 
        __host__ virtual bool Serialise(Json::Node& rootNode) { return false; }
        __host__ virtual bool Deserialise(const Json::Node& rootNode) { return false; }

    protected:
        __host__ Serialisable() {} 
    };
}