#pragma once

#include "thirdparty/rapidjson/document.h"
#include "thirdparty/rapidjson/allocators.h"
#include "thirdparty/rapidjson/error/en.h"
#include "StdIncludes.h"
 
namespace Json
{
    class Node
    {
    public:
        Node();

    private:

    };

    class Document : public Node
    {
    public:
        Document();

        void Parse(const std::string& data);

    private:
        rapidjson::Document m_document;
    };
}