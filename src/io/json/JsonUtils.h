#pragma once

#include "thirdparty/rapidjson/document.h"
#include "thirdparty/rapidjson/allocators.h"
#include "thirdparty/rapidjson/error/en.h"
#include "core/StringUtils.h"
#include "io/Log.h"
#include "core/Assert.h"
#include "core/Types.h"
#include <map>
#include <functional>

namespace Enso
{
    namespace Json
    {
        enum RequiredFlags : uint
        {
            kSilent             = 1,
            kRequiredWarn       = 2,
            kRequiredAssert     = 4,
            kNotBlank           = 8,
            kPathIsDAG          = 16
        };

        enum KeyFormat : int
        {
            kKeyInvalid = -1,
            kKeyName = 0,
            kKeyDAG = 1
        };

        template<typename... Pack>
        inline void ReportError(const uint flags, const std::string message, Pack... pack)
        {
            if (flags & kRequiredWarn) { Log::Warning("Warning: " + message, pack...); }
            else if (flags & kRequiredAssert) { AssertMsgFmt(false, ("Error: " + message).c_str(), pack...); }
        }

        inline void ReportError(const uint flags, const std::string message)
        {
            if (flags & kRequiredWarn) { Log::Warning("Warning: " + message); }
            else if (flags & kRequiredAssert) { AssertMsg(false, ("Error: " + message).c_str()); }
        }

        class Document;

        class Node
        {
        protected:
            rapidjson::Value*                       m_node;
            const Document*                         m_rootDocument;
            rapidjson::Document::AllocatorType*     m_allocator;
            std::string                             m_dagPath;

        public:
            Node() : m_node(nullptr), m_rootDocument(nullptr), m_allocator(nullptr) {}
            Node(const std::nullptr_t&) : m_node(nullptr), m_rootDocument(nullptr), m_allocator(nullptr) {}
            Node(rapidjson::Value* node, const Node& parent) : m_node(node), m_rootDocument(parent.m_rootDocument), m_allocator(parent.m_allocator) {}
            Node(rapidjson::Value* node, const Node& parent, const ::std::string& id) :
                m_node(node), m_rootDocument(parent.m_rootDocument), m_allocator(parent.m_allocator)
            {
                m_dagPath = (parent.m_dagPath.empty()) ? id : (parent.m_dagPath + kDAGDelimiter + id);
            }
            Node(const Node&) = default;
            Node& operator=(const Node&) = default;
            //Node(Node&&);
            //Node& operator=(Node&&);

            rapidjson::Value* GetChildImpl(const std::string& path, uint flags) const;

        public:
            static const char                       kDAGDelimiter = '/';

            template<bool IsConst>
            class __Iterator
            {
            public:
                __Iterator(rapidjson::Value::MemberIterator& it, const Node& parentNode) : m_it(it), m_parentNode(parentNode) {}

                inline __Iterator& operator++() { ++m_it; return *this; }

                template<bool C = IsConst>
                inline typename std::enable_if<!C, Node>::type operator*() const { return Node(&(m_it->value), m_parentNode, m_it->name.GetString()); }
                template<bool C = IsConst>
                inline typename std::enable_if<C, const Node>::type operator*() const { return Node(&(m_it->value), m_parentNode, m_it->name.GetString()); }

                inline bool operator!=(const __Iterator& other) const { return m_it != other.m_it; }

                inline std::string Name() const { return m_it->name.GetString(); }
                inline const rapidjson::Value& Value() const { return m_it->value; }

            private:
                rapidjson::Value::MemberIterator     m_it;
                const Node& m_parentNode;
            };

            using Iterator = __Iterator<false>;
            using ConstIterator = __Iterator<true>;

        private:
            template<bool/* = is_arithmetic<T>*/, bool/* = is_integral<T>*/> struct GetValueImpl
            {
                template<typename T> static void f(const rapidjson::Value& node, const std::string& name, T& value, const uint flags)
                {
                    // NOTE: An error here means that a value is being requested that hasn't explicitly been specialised e.g. GetValue("", var) where var is a vector. 
                    AssertMsgFmt(node.IsString(), "Attribute '%s' is not of type string.", name.c_str());
                    value = node.GetString();

                    if (value.empty() && flags & kNotBlank)
                    {
                        ReportError(flags, "Attribute '%s' must not be blank.", name);
                    }
                }
            };
         
            // Ensures that the specified DAG path exists up until the last element (which will be decided by the calling function)
            Node ResolveDAGRoute(const std::string& path, const uint flags, rapidjson::Value* node, std::function<Node(rapidjson::Value*, const std::string&, const uint)> functor);

        public:
            inline void CheckOk() const { AssertMsg(m_node && m_allocator, "Invalid or unitialised JSON node."); }
            bool IsValidName(const std::string& name) const; 
            int GetKeyFormat(const std::string& path) const;

            const Document& GetRootDocument() const { Assert(m_rootDocument); return *m_rootDocument; }
            void DeepCopy(const Node& other);

            inline bool HasDAGPath() const { return !m_dagPath.empty(); }
            inline const std::string& GetDAGPath() const { return m_dagPath; }

            std::string Stringify(const bool pretty) const;

            Iterator begin() { CheckOk(); return Iterator(m_node->MemberBegin(), *this); }
            Iterator end() { CheckOk(); return Iterator(m_node->MemberEnd(), *this); }
            ConstIterator begin() const { CheckOk(); return ConstIterator(m_node->MemberBegin(), *this); }
            ConstIterator end() const { CheckOk(); return ConstIterator(m_node->MemberEnd(), *this); }

            bool IsObject() const;
            inline int NumMembers() const { return Size(); }
            int Size() const;

            template<typename T>
            void AddValue(const std::string& path, const T& value, const uint flags = 0)
            {
                auto functor = [this, &value](rapidjson::Value* node, const std::string& name, const uint flags) -> Node
                {
                    node->AddMember(rapidjson::Value(name.c_str(), *m_allocator).Move(), value, *m_allocator);
                    return Node();
                };
                ResolveDAGRoute(path, flags, m_node, functor);
            }

            void AddValue(const std::string& name, const std::string& value, const uint flags = 0);

            template<typename VecType>
            void AddVector(const std::string& name, const VecType& vec, const uint flags = 0)
            {
                std::vector<typename VecType::kType> values(VecType::kDims);
                for (int i = 0; i < VecType::kDims; i++) { values[i] = vec[i]; }

                AddArray(name, values, flags);
            }

            template<typename T>
            void AddArray(const std::string& path, const std::vector<T>& values, const uint flags = 0)
            {
                auto functor = [this, &values](rapidjson::Value* node, const std::string& name, const uint flags) -> Node
                {
                    rapidjson::Value jsonArray(rapidjson::kArrayType);
                    for (const auto& element : values)
                    {
                        jsonArray.PushBack(element, *m_allocator);
                    }
                    node->AddMember(rapidjson::Value(name.c_str(), *m_allocator).Move(), jsonArray, *m_allocator);
                    return Node();
                };
                ResolveDAGRoute(path, flags, m_node, functor);
            }

            void AddArray(const std::string& name, const std::vector<std::string>& values, const uint flags = 0);

            template<typename T>
            void AddArray2D(const std::string& path, const std::vector<std::vector<T>>& values, const uint flags = 0)
            {
                static_assert(std::is_arithmetic<T>::value, "Can only add arithmetic values to 2D arrays for now.");

                auto functor = [this, &values](rapidjson::Value* node, const std::string& name, const uint flags) -> Node
                {
                    rapidjson::Value rowArray(rapidjson::kArrayType);
                    for (const auto& row : values)
                    {
                        rapidjson::Value colArray(rapidjson::kArrayType);
                        for (const auto& col : row)
                        {
                            colArray.PushBack(col, *m_allocator);
                        }
                        rowArray.PushBack(colArray, *m_allocator);
                    }
                    node->AddMember(rapidjson::Value(name.c_str(), *m_allocator).Move(), rowArray, *m_allocator);
                    return Node();
                };
                ResolveDAGRoute(path, flags, m_node, functor);               
            }

            template<typename T>
            void AddEnumeratedParameter(const std::string& path, const std::vector<std::string>& ids, const T& parameterValue, const uint flags = 0)
            {
                AssertMsgFmt(parameterValue >= 0 && parameterValue < ids.size(),
                    "Parameter value %i for attribute '%s' is not in the valid range [0, %i)", parameterValue, path.c_str(), ids.size());

                auto functor = [this, &parameterValue, &ids](rapidjson::Value* node, const std::string& name, const uint flags) -> Node
                {
                    node->AddMember(rapidjson::Value(name.c_str(), *m_allocator).Move(),
                                    rapidjson::Value(ids[parameterValue].c_str(), *m_allocator).Move(), *m_allocator);
                    return Node();
                };
                ResolveDAGRoute(path, flags, m_node, functor);                
            }

            Node AddChildObject(const std::string& name, const int flags = 0);

            bool GetBool(const std::string& name, const bool defaultValue, const uint flags) const;

            template<typename T>
            bool GetValue(const std::string& name, T& value, const uint flags) const
            {
                CheckOk();
                const rapidjson::Value* child = GetChildImpl(name, flags);
                if (!child) { return false; }

                AssertMsgFmt(!child->IsObject() && !child->IsArray(), "Value at '%s' is not a scalar.", name.c_str());

                GetValueImpl<std::is_arithmetic<T>::value, std::is_integral<T>::value>::f(*child, name, value, flags);
                return true;
            }

            const Node GetChild(const std::string& name, const uint flags) const;
            const Node GetChildObject(const std::string& name, const uint flags) const;
            const Node GetChildArray(const std::string& name, const uint flags) const;

            template<typename Type>
            bool GetArrayValues(const std::string& name, std::vector<Type>& values, const uint flags) const
            {
                const Node child = GetChildArray(name, flags);
                if (!child) { return false; }
                rapidjson::Value& array = *child.m_node;

                values.resize(array.Size());
                for (size_t idx = 0; idx < array.Size(); idx++)
                {
                    Type value;
                    GetValueImpl< std::is_arithmetic<Type>::value,
                        std::is_integral<Type>::value >::f(array[idx],
                            "[unknown; getVector()]",
                            value, flags);
                    values[idx] = value;
                }
                return true;
            }

            template<typename Type>
            bool GetArray2DValues(const std::string& name, std::vector<std::vector<Type>>& values, const uint flags) const
            {
                const Node child = GetChildArray(name, flags);
                if (!child) { return false; }
                rapidjson::Value& rowArray = *child.m_node;

                if (!rowArray.IsArray())
                {
                    ReportError(flags, "Error: JSON 2D array '%s' expected objects as elements.", name.c_str());
                    return false;
                }

                values.resize(rowArray.Size());

                for (size_t row = 0; row < rowArray.Size(); row++)
                {
                    if (!rowArray[row].IsArray())
                    {
                        ReportError(flags, "Error: JSON 2D array '%s' expected objects as elements.", name.c_str());
                        return false;
                    }

                    rapidjson::Value& colArray = rowArray[row];
                    values[row].reserve(colArray.Size());

                    for (size_t col = 0; col < colArray.Size(); col++)
                    {
                        Type value;
                        GetValueImpl< std::is_arithmetic<Type>::value,
                            std::is_integral<Type>::value >::f(colArray[col],
                                "[unknown; getVector()]",
                                value, flags);
                        values[row].emplace_back(value);
                    }
                }
                return true;
            }

            template<typename VecType>
            bool GetVector(const std::string& name, VecType& vec, const uint flags) const
            {
                std::vector<typename VecType::kType> values;
                if (!GetArrayValues(name, values, flags)) { return false; }

                if (VecType::kDims != values.size())
                {
                    ReportError(flags, "JSON array '%s' expects % i elements.", name, VecType::kDims);
                    return false;
                }

                for (int i = 0; i < values.size(); i++) { vec[i] = values[i]; }
                return true;
            }

            bool GetEnumeratedParameter(const std::string& parameterName, const std::vector<std::string>& ids, int& parameterValue, const uint flags) const;
            bool GetEnumeratedParameter(const std::string& parameterName, const std::map<std::string, int>& map, int& parameterValue, const uint flags) const;

            inline operator bool() const { return m_node; }
            inline bool operator!() const { return !m_node; }

            inline rapidjson::Value* GetPtr() { return m_node; }
        };

        class Document : public Node
        {
        public:
            Document()
            {
                m_document.SetObject();
                m_node = &m_document;
                m_allocator = &m_document.GetAllocator();
                m_rootDocument = this;
            }

            ~Document() = default;
            Document(const Document&) = delete;
            Document(const Document&&) = delete;

            Document(const std::string& str) : Document()
            {
                Parse(str);
            }

            Document& operator=(const Node& node)
            {
                m_allocator->Clear();
                DeepCopy(node);
                return *this;
            }

            Document& operator=(const nullptr_t& node)
            {
                m_allocator->Clear();
                return *this;
            }

            inline Document& operator=(const Document& node) { return this->operator=(static_cast<const Node&>(node)); }

            void Clear()
            {
                m_document.SetObject();
                m_allocator->Clear();
            }

            void Parse(const std::string& data);
            void Deserialise(const std::string& filePath);
            void Serialise(const std::string& filePath);

            const std::string& GetOriginFilePath() const { return m_originFilePath; }

        private:
            rapidjson::Document         m_document;

            std::string                 m_originFilePath;
        };

        template<> struct Node::GetValueImpl<true, true>
        {
            template<typename T> static void f(const rapidjson::Value& node, const std::string& name, T& value, const uint flags)
            {
                AssertMsgFmt(node.IsInt() || node.IsInt64(), "Node '%s' is not an integer type.", name.c_str());
                value = T(node.GetInt64());
            }

            static void f(const rapidjson::Value& node, const std::string& name, bool& value, const uint flags)
            {
                value = node.GetBool();
            }
        };

        template<> struct Node::GetValueImpl<true, false>
        {
            template<typename T> static void f(const rapidjson::Value& node, const std::string& name, T& value, const uint flags)
            {
                //PT_ASSERT_DETAILED(node.IsDouble()/* || node.IsFloat()*/,
                //                     "Node '%s' is not a floating point type.", name.c_str());
                value = T(node.GetDouble());
            }
        };
    }
}