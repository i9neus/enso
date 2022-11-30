#pragma once

#include "core/Assert.h"

namespace Enso
{
    namespace Json { class Node; }
    
    // Use upper 16 bits so we don't clash with JSON parser flags
    enum SerialisableFlags : int
    {
        // Only de/serialise the object ID
        kSerialiseIdOnly = 1 << 16,
        // Only de/serialise the parameters exposed to the UI
        kSerialiseExposedOnly = 1 << 17,
        // De/serialise everything
        kSerialiseAll = 1 << 18
    };

    class Serialisable
    {
    public:
        __host__ virtual bool Serialise(Json::Node& rootNode, const int flags) const { return false; }
        __host__ virtual uint Deserialise(const Json::Node& rootNode, const int flags) { return 0u; }

    private:
    };
}