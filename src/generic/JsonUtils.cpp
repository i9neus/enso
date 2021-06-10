#include "JsonUtils.h"
#include "StringUtils.h"

#include "thirdparty/rapidjson/stringbuffer.h"
#include "thirdparty/rapidjson/prettywriter.h"
 
namespace Json
{
    rapidjson::Value* Node::GetChildImpl(const std::string& path, bool required) const
    {
        Assert(m_node);
        AssertMsg(!path.empty(), "Must specify a path to a node.");
        AssertMsg(m_node->IsObject(), "Parent node is not an object.");

        Lexer lex(path);
        std::string parentID = "[root node]";
        rapidjson::Value* node = m_node;
        while (lex)
        {
            std::string childID;
            bool success = lex.ParseToken(childID, [](char c) { return std::isalnum(c) || c == '_'; });

            AssertMsgFmt(success, "Malformed or missing identifier in path string '%s'.", path.c_str());

            if (!node->IsObject())
            {
                AssertMsgFmt(!required, "A required JSON node '%s' is invalid ('%s' is not an object.)",
                    path.c_str(), parentID.c_str());
                return nullptr;
            }

            rapidjson::Value::MemberIterator jsonIt = node->FindMember(childID.c_str());
            if (jsonIt == node->MemberEnd())
            {
                AssertMsgFmt(!required, "A required JSON node '%s' is invalid ('%s' not found)",
                    path.c_str(), childID.c_str());
                return nullptr;
            }

            node = &jsonIt->value; // Jump into the child node

            if (!lex || !lex.PeekNext('.')) { return node; }

            parentID = childID;
        }

        AssertMsg(false, "Shouldn't be here!");
        return nullptr;
    }

    void Document::DeepCopy(const Document& other)
    {
        m_document.SetObject();
        m_document.CopyFrom(other.m_document, m_document.GetAllocator());
    }

    void Document::Parse(const std::string& data)
    {
        AssertMsg(!data.empty(), "Bad data.");

        m_document.Parse(data.c_str());

        if (m_document.HasParseError())
        {
            int32_t lineNumber = 1;
            for (size_t idx = 0; idx < data.length() && idx <= m_document.GetErrorOffset(); idx++)
            {
                if (data[idx] == '\n') { lineNumber++; }
            }

            AssertMsgFmt(false, "RapidJson error: %s (line %i)", rapidjson::GetParseError_En(m_document.GetParseError()), lineNumber);
        }

        m_node = &m_document;
        m_allocator = &m_document.GetAllocator();
    }

    std::string Document::LoadTextFile(const std::string& filePath) const
    {
        std::ifstream file(filePath, std::ios::in);

        AssertMsgFmt(file.good(), "Couldn't open file '%s'", filePath.c_str());

        const int32_t fileSize = file.tellg();
        std::string data;
        data.reserve(fileSize + 1);

        data.assign(std::istreambuf_iterator<char>(file),
            std::istreambuf_iterator<char>());

        file.close();
        return data;
    }

    void Document::Load(const std::string& filePath)
    {
        std::string raw = LoadTextFile(filePath);

        // Erase commented-out blocks
        for (int i = 0; i < raw.size() && raw[i] != '\0'; i++)
        {
            if (raw[i] == '/' && i < raw.size() - 1 && raw[i + 1] == '/')
            {
                int j;
                for (j = i; j < raw.size() && raw[j] != '\0' && raw[j] != '\n'; j++)
                {
                    raw[j] = ' ';
                }
                i = j;
            }
        }

        Parse(raw);
    }

    std::string Document::Stringify()
    {
        rapidjson::StringBuffer buffer;
        rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
        m_document.Accept(writer);

        return buffer.GetString();
    }
}