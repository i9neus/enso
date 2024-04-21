#include "JsonUtils.h"

#include "thirdparty/rapidjson/stringbuffer.h"
#include "thirdparty/rapidjson/prettywriter.h"

#include "io/FilesystemUtils.h"
#include <functional>

#undef GetObject
 
namespace Enso
{
    namespace Json
    {
        /*Node::Node(Node&& other)
        {
            m_node = other.m_node;
            m_rootDocument = other.m_rootDocument;
            m_allocator = m_allocator;
            m_dagPath = std::move(other.m_dagPath);
        }

        Node& Node::operator=(Node&& other)
        {
            m_node = other.m_node;
            m_rootDocument = other.m_rootDocument;
            m_allocator = m_allocator;
            m_dagPath = std::move(other.m_dagPath);
            return *this;
        }*/

        int Node::GetKeyFormat(const std::string& name) const
        {
            // JSON node names may only contain alphanumeric characters, underscores, dashes, and the DAG delimiter
            if (name.empty()) { return false; }
            int keyFormat = kKeyName;
            for (auto& c : name)
            {
                if (!std::isalnum(c) && c != '_' && c != '-') { return kKeyInvalid; }
                else if (c == kDAGDelimiter) { keyFormat = kKeyDAG; }
            }
            return keyFormat;
        }
        
        bool Node::IsValidName(const std::string& name) const
        {
            // JSON node names may only contain alphanumeric characters, underscores and dashes
            if (name.empty()) { return false; }
            for (auto& c : name)
            {
                if (!std::isalnum(c) && c != '_' && c != '-') { return false; }
            }
            return true;
        }
        
        rapidjson::Value* Node::GetChildImpl(const std::string& path, const uint flags) const
        {
            Assert(m_node);
            AssertMsg(!path.empty(), "Must specify a path to a node.");
            AssertMsg(m_node->IsObject(), "Parent node is not an object.");

            // If this ID isn't a DAG, search for it directly without trying to parse it as a path
            if (!(flags & kPathIsDAG))
            {
                if (!m_node->IsObject())
                {
                    ReportError(flags, "JSON node '%s' is invalid ('%s' is not an object.)", path.c_str(), path.c_str());
                    return nullptr;
                }

                rapidjson::Value::MemberIterator jsonIt = m_node->FindMember(path.c_str());
                if (jsonIt == m_node->MemberEnd())
                {
                    ReportError(flags, "JSON node literal '%s' is invalid ('%s' not found)", path.c_str(), path.c_str());
                    return nullptr;
                }
                return &jsonIt->value;
            }

            // Otherwise, break it up using the lexer and extract each component in turn
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
                    ReportError(flags, "JSON node '%s' is invalid ('%s' is not an object.)", path.c_str(), parentID.c_str());
                    return nullptr;
                }

                rapidjson::Value::MemberIterator jsonIt = node->FindMember(childID.c_str());
                if (jsonIt == node->MemberEnd())
                {
                    ReportError(flags, "JSON node '%s' is invalid ('%s' not found)", path.c_str(), childID.c_str());
                    return nullptr;
                }

                node = &jsonIt->value; // Jump into the child node

                if (!lex || !lex.PeekNext(kDAGDelimiter)) { return node; }

                parentID = childID;
            }

            AssertMsgFmt(false, "Encountered improperly formatted JSON path '%s'", path.c_str());
            return nullptr;
        }

        void Node::AddValue(const std::string& path, const std::string& value, const uint flags)
        {
            auto functor = [this, &value](rapidjson::Value* node, const std::string& name, const uint flags) -> Node
            {
                node->AddMember(rapidjson::Value(name.c_str(), *m_allocator).Move(),
                                  rapidjson::Value(value.c_str(), *m_allocator).Move(), *m_allocator);

                return Node();
            };

            ResolveDAGRoute(path, flags, m_node, functor);
        }

        void Node::AddArray(const std::string& path, const std::vector<std::string>& values, const uint flags)
        {
            auto functor = [this, &values](rapidjson::Value* node, const std::string& name, const uint flags) -> Node
            {
                rapidjson::Value jsonArray(rapidjson::kArrayType);
                for (const auto& element : values)
                {
                    jsonArray.PushBack(rapidjson::Value(element.c_str(), *m_allocator).Move(), *m_allocator);
                }
                node->AddMember(rapidjson::Value(name.c_str(), *m_allocator).Move(), jsonArray, *m_allocator);
                return Node();
            };

            ResolveDAGRoute(path, flags, m_node, functor);
        }

        Node Node::ResolveDAGRoute(const std::string& path, const uint flags, rapidjson::Value* node, std::function<Node(rapidjson::Value*, const std::string&, const uint)> functor)
        {
            CheckOk();
            
            if (!(flags & kPathIsDAG))
            {
                // Verify that the key is correctly formatted
                const int keyFormat = GetKeyFormat(path);
                AssertMsgFmt(keyFormat != kKeyInvalid, "'%s' is not a valid node name (must be alphanumeric, underscore or dash)", path.c_str());
                AssertMsgFmt(keyFormat != kKeyDAG, "'%s' is formatted like a DAG, however the kPathIsDAG flag was not specified.", path.c_str());
                    
                return functor(node, path, flags);
            }
            
            // Parse the DAG path into its constituent tokens
            Lexer lex(path);
            Assert(lex);
            std::vector<std::string> tokens;
            do
            {
                tokens.resize(tokens.size() + 1);
                bool success = lex.ParseToken(tokens.back(), [](char c) { return c != kDAGDelimiter; });
                AssertMsgFmt(success, "Malformed or missing identifier in path string '%s'.", path.c_str());
                AssertMsgFmt(IsValidName(tokens.back()), "Token '%s' in DAG path '%s' is not a valid node name.", tokens.back().c_str(), path.c_str());

            } while (lex && lex.PeekNext(kDAGDelimiter));

            // Create a path into the tree according to the token list, creating new nodes as necessary
            for (int idx = 0; idx < tokens.size() - 1; ++idx)
            {
                // If a child with this ID does not already exist,  create it
                rapidjson::Value::MemberIterator jsonIt = node->FindMember(tokens[idx].c_str());
                if (jsonIt == node->MemberEnd())
                {
                    node->AddMember(rapidjson::Value(tokens[idx].c_str(), *m_allocator).Move(), rapidjson::Value().Move(), *m_allocator);
                    node = &(*node)[tokens[idx].c_str()];
                    node->SetObject();
                }
                // Otherwise, jump into the existing node
                else
                {
                    AssertMsgFmt(jsonIt->value.IsObject(), "Node '%s' in DAG '%s' is not an object.", tokens[idx].c_str(), path.c_str());
                    node = &jsonIt->value;
                }
            }

            return functor(node, tokens.back(), flags);
        }

        Node Node::AddChildObject(const std::string& path, const int flags)
        {
            auto functor = [this](rapidjson::Value* node, const std::string& name, const uint flags) -> Node
            {
                // Add the new child node
                node->AddMember(rapidjson::Value(name.c_str(), *m_allocator).Move(), rapidjson::Value().Move(), *m_allocator);
                rapidjson::Value& newNode = (*node)[name.c_str()];
                newNode.SetObject();

                return Node(&newNode, *this);
            };

            return ResolveDAGRoute(path, flags, m_node, functor);
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
                ReportError(flags, "Node '%s' is not an array.", name.c_str());
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
                ReportError(flags, "JSON node/attribute '%s' was expected but not found\n", parameterName);
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
            // FIXME: SetObject does not automatically deallocate memory that may have been allocated to store nodes already at this location. 
            // Deallocation must be done by manually calling Clear() on the allocator object. This is a fundamental problem with rapidJson
            // so consider swapping it out with a better JSON library. e.g. https://github.com/nlohmann/json
            m_node->SetObject();
            m_node->CopyFrom(*other.m_node, *m_allocator);
        }

        std::string Node::Stringify(const bool pretty) const
        {
            if (!m_node) { return ""; }

            rapidjson::StringBuffer buffer;
            if (pretty)
            {
                rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
                m_node->Accept(writer);
            }
            else
            {
                rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
                m_node->Accept(writer);
            }

            return buffer.GetString();
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

        void Document::Deserialise(const std::string& filePath)
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

        void Document::Serialise(const std::string& filePath)
        {
            WriteTextFile(filePath, Stringify(true));
        }

        bool Node::IsObject() const { CheckOk(); return m_node->IsObject(); }

        int Node::Size() const
        {
            CheckOk();
            if (!m_node->IsObject()) { return 0; }

            return m_node->GetObject().MemberCount();
        }
    }
}