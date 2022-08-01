#pragma once

#include "thirdparty/rapidjson/document.h"
#include "thirdparty/rapidjson/allocators.h"
#include "thirdparty/rapidjson/error/en.h"
#include "StdIncludes.h"
#include "StringUtils.h"
#include "Log.h"
#include <map>

namespace Json
{        
    enum RequiredFlags : uint 
    { 
        kSilent =           1 << 0,
        kRequiredWarn =     1 << 1,
        kRequiredAssert =   1 << 2,
        kNotBlank =         1 << 3,
        kLiteralID =        1 << 4
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
        rapidjson::Value* m_node;
        const Document* m_rootDocument;
        rapidjson::Document::AllocatorType* m_allocator;
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

        template<typename T, typename Lambda>
        void AddArrayImpl(const std::string& name, const std::vector<T>& values, Lambda Push)
        {
            CheckOk();
            rapidjson::Value jsonArray(rapidjson::kArrayType);
            for (const auto& element : values) { Push(jsonArray, element); }

            m_node->AddMember(rapidjson::Value(name.c_str(), *m_allocator).Move(), jsonArray, *m_allocator);
        }

    public:
        inline void CheckOk() const { AssertMsg(m_node && m_allocator, "Invalid or unitialised JSON node."); }
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
        int NumMembers() const;

        template<typename T>
        void AddValue(const std::string& name, const T& value)
        {
            CheckOk();
            m_node->AddMember(rapidjson::Value(name.c_str(), *m_allocator).Move(), value, *m_allocator);
        }

        void AddValue(const std::string& name, const std::string& value);

        template<typename VecType>
        void AddVector(const std::string& name, const VecType& vec)
        {
            std::vector<typename VecType::kType> values(VecType::kDims);
            for (int i = 0; i < VecType::kDims; i++) { values[i] = vec[i]; }            
            
            AddArray(name, values);
        }

        template<typename T>
        void AddArray(const std::string& name, const std::vector<T>& values)
        {
            AddArrayImpl(name, values, [&](rapidjson::Value& jsonArray, const T& element) { jsonArray.PushBack(element, *m_allocator); });
        }

        void AddArray(const std::string& name, const std::vector<std::string>& values);

        template<typename T>
        void AddArray2D(const std::string& name, const std::vector<std::vector<T>>& values)
        {
            static_assert(std::is_arithmetic<T>::value, "Can only add arithmetic values to 2D arrays for now.");
            
            CheckOk();
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

            m_node->AddMember(rapidjson::Value(name.c_str(), *m_allocator).Move(), rowArray, *m_allocator);     
        }

        template<typename T>
        void AddEnumeratedParameter(const std::string& parameterName, const std::vector<std::string>& ids, const T& parameterValue)
        {
            AssertMsgFmt(parameterValue >= 0 && parameterValue < ids.size(), 
                "Parameter value %i for attribute '%s' is not in the valid range [0, %i)", parameterValue, parameterName.c_str(), ids.size());
            
            CheckOk();
            m_node->AddMember(rapidjson::Value(parameterName.c_str(), *m_allocator).Move(),
                rapidjson::Value(ids[parameterValue].c_str(), *m_allocator).Move(), *m_allocator);
        }

        const Node AddChildObject(const std::string& name);
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

        Document& operator=(const Node& node)
        {
            m_allocator->Clear();
            DeepCopy(node);
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
        rapidjson::Document m_document;

        std::string m_originFilePath;
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