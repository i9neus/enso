#pragma once

#include "core/Asset.cuh"

namespace Enso
{
    namespace Json
    {
        class Node;
        class Document;
    }
    
    class Serialisable
    {
    public: 
        __host__ virtual bool Serialise(Json::Node& rootNode) { return false; }
        __host__ virtual bool Deserialise(const Json::Node& rootNode) { return false; }

    protected:
        __host__ Serialisable() {} 
    };
}