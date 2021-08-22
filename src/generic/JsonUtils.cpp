#include "JsonUtils.h"

#include "thirdparty/rapidjson/stringbuffer.h"
#include "thirdparty/rapidjson/prettywriter.h"

#include "generic/FilesystemUtils.h"

#undef GetObject
 
namespace Json
{
    rapidjson::Value* Node::GetChildImpl(const std::string& path, const uint flags) const
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
            bool success = lex.ParseToken(childID, [](char c) { return c != kDAGDelimiter; });

            AssertMsgFmt(success, "Malformed or missing identifier in path string '%s'.", path.c_str());

            if (!node->IsObject())
            {                
                AssertMsgFmt(flags != kRequiredAssert, "A required JSON node '%s' is invalid ('%s' is not an object.)",
                    path.c_str(), parentID.c_str());
                
                if (flags == kRequiredWarn)
                {
                    Log::Warning("JSON node/attribute '%s' was expected but not found ('%s' is not an object.)\n", path, parentID);
                }

                return nullptr;
            }

            rapidjson::Value::MemberIterator jsonIt = node->FindMember(childID.c_str());
            if (jsonIt == node->MemberEnd())
            {
                AssertMsgFmt(flags != kRequiredAssert, "A required JSON node '%s' is invalid ('%s' not found)",
                    path.c_str(), childID.c_str());

                if (flags == kRequiredWarn)
                {
                    Log::Warning("JSON node/attribute '%s' was expected but not found.\n", path);
                }

                return nullptr;
            }

            node = &jsonIt->value; // Jump into the child node

            if (!lex || !lex.PeekNext(kDAGDelimiter)) { return node; }

            parentID = childID;
        } 

        AssertMsg(false, "Shouldn't be here!");
        return nullptr;
    }

    void Node::AddValue(const std::string& name, const std::string& value)
    {
        CheckOk();
        m_node->AddMember(rapidjson::Value(name.c_str(), *m_allocator).Move(),
            rapidjson::Value(value.c_str(), *m_allocator).Move(), *m_allocator);
    }

    void Node::AddArray(const std::string& name, const std::vector<std::string>& values)
    {
        AddArrayImpl(name, values, [&](rapidjson::Value& jsonArray, const std::string& element)
            {
                jsonArray.PushBack(rapidjson::Value(element.c_str(), *m_allocator).Move(), *m_allocator);
            });
    }

    const Node Node::AddChildObject(const std::string& name)
    {
        CheckOk();
        m_node->AddMember(rapidjson::Value(name.c_str(), *m_allocator).Move(), rapidjson::Value().Move(), *m_allocator);
        rapidjson::Value& newNode = (*m_node)[name.c_str()];
        newNode.SetObject();
        return Node(&newNode, *this);
    }

    bool Node::GetBool(const std::string& name, const bool defaultValue, const uint flags) const
    {
        bool value = defaultValue;
        GetValue(name, value, flags);
        return value;
    }

    const Node Node::GetChild(const std::string& name, const uint flags) const
    {
        CheckOk();
        return Node(GetChildImpl(name, flags), *this, name);
    }

    const Node Node::GetChildObject(const std::string& name, const uint flags) const
    {
        CheckOk();
        rapidjson::Value* child = GetChildImpl(name, flags);
        if (!child || !child->IsObject()) { return nullptr; }

        return Node(child, *this, name);
    }

    const Node Node::GetChildArray(const std::string& name, const uint flags) const
    {
        rapidjson::Value* child = GetChildImpl(name, flags);
        if (!child) { return Node(); }

        if (!child->IsArray())
        {
            AssertMsgFmt(flags != kRequiredAssert, "Node '%s' is not an array.", name.c_str());
            return Node();
        }

        return Node(child, *this, name);
    }

    bool Node::GetEnumeratedParameter(const std::string& parameterName, const std::vector<std::string>& ids, int& parameterValue, const uint flags) const
    {
        std::map<std::string, int> map;
        for (int idx = 0; idx < ids.size(); idx++)
        {
            map.insert(std::make_pair(Lowercase(ids[idx]), idx));
        }
        return GetEnumeratedParameter(parameterName, map, parameterValue, flags);
    }

    bool Node::GetEnumeratedParameter(const std::string& parameterName, const std::map<std::string, int>& map, int& parameterValue, const uint flags) const
    {
        AssertMsg(!map.empty(), "Empty enumeration map.");

        std::string parameterStr;
        if (!GetValue(parameterName, parameterStr, flags)) { return false; }

        if (parameterStr.empty())
        {
            AssertMsgFmt(flags != kRequiredAssert, "Required parameter '%s' not specified.", parameterName.c_str());
            if (flags == kRequiredWarn)
            {
                Log::Warning("JSON node/attribute '%s' was expected but not found\n", parameterName);
            }
            return false;
        }

        MakeLowercase(parameterStr);

        auto it = map.find(parameterStr);
        if (it != map.end())
        {
            parameterValue = it->second;
            return true;
        }
        else
        {
            // Not found, so print an error with the list of possible options
            std::string error = tfm::format("Invalid value for parameter '%s'. Options are:", parameterName);
            for (const auto& element : map)
            {
                error += tfm::format(" '%s'", element.first);
            }

            AssertMsgFmt(flags != kRequiredAssert, error.c_str());
            if (flags == kRequiredWarn)
            {
                Log::Warning(error);
            }
            return false;
        }

        return true;
    }

    void Node::DeepCopy(const Node& other)
    {
        m_node->SetObject();
        m_node->CopyFrom(*other.m_node, *m_allocator);
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

    void Document::Load(const std::string& filePath)
    {
        m_originFilePath = filePath;
        std::string raw = ReadTextFile(filePath);

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

    void Document::WriteFile(const std::string& filePath)
    {
        WriteTextFile(filePath, Stringify());
    }

    std::string Document::Stringify()
    {
        rapidjson::StringBuffer buffer;
        rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
        m_document.Accept(writer);

        return buffer.GetString();
    }

    bool Node::IsObject() const { CheckOk(); return m_node->IsObject(); }

    int Node::NumMembers() const
    {
        CheckOk();
        if (!m_node->IsObject()) { return 0; }

        return m_node->GetObject().MemberCount();
    }
}