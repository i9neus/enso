#pragma once

#include <map>
#include <unordered_map>

namespace Enso
{

#define ASSOCIATED_CONTAINER_CONTAINS(ContainerType) template<typename T, typename S> \
    inline bool Contains(const ContainerType<T, S>& map, const T& key) \
    { \
        return map.find(key) != map.end(); \
    } 

    ASSOCIATED_CONTAINER_CONTAINS(std::map)
    ASSOCIATED_CONTAINER_CONTAINS(std::multimap)
    ASSOCIATED_CONTAINER_CONTAINS(std::unordered_map)
    ASSOCIATED_CONTAINER_CONTAINS(std::unordered_multimap)

}