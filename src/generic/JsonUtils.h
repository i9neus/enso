#pragma once

#include "thirdparty/rapidjson/document.h"
#include "thirdparty/rapidjson/allocators.h"
#include "thirdparty/rapidjson/error/en.h"
#include "StdIncludes.h"
#include "Log.h"

namespace Json
{        
    enum RequiredFlags : uint { kSilent, kRequiredWarn, kRequiredAssert };
    
    class Node
    {
    protected:
        rapidjson::Value*                       m_node;
        rapidjson::Document::AllocatorType*     m_allocator;
        std::string                             m_dagPath;
        static const char                       kDAGDelimiter = '/';

        Node() : m_node(nullptr), m_allocator(nullptr) {}
        Node(const std::nullptr_t&) : m_node(nullptr), m_allocator(nullptr) {}
        Node(rapidjson::Value* node, const Node& parent) : m_node(node), m_allocator(parent.m_allocator) {}
        Node(rapidjson::Value* node, const Node& parent, const ::std::string& id) : 
            m_node(node), m_allocator(parent.m_allocator)
        {
            m_dagPath = (parent.m_dagPath.empty()) ? id : (parent.m_dagPath + kDAGDelimiter + id);
        }

        rapidjson::Value* GetChildImpl(const std::string& path, uint flags) const;

    public:    
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

        private:
            rapidjson::Value::MemberIterator     m_it;
            const Node&                          m_parentNode;
        };
        
        using Iterator = __Iterator<false>;
        using ConstIterator = __Iterator<true>;

    private:
        template<bool/* = is_arithmetic<T>*/, bool/* = is_integral<T>*/> struct GetValueImpl
        {
            template<typename T> static void f(const rapidjson::Value& node, const std::string& name, T& value)
            {
                // NOTE: An error here means that a value is being requested that hasn't explicitly been specialised e.g. GetValue("", var) where var is a vector. 
                AssertMsgFmt(node.IsString(), "Node '%s' is not of type string.", name.c_str());
                value = node.GetString();
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
        
        inline bool HasDAGPath() const { return !m_dagPath.empty(); }
        inline const std::string& GetDAGPath() const { return m_dagPath; }

        Iterator begin() { CheckOk(); return Iterator(m_node->MemberBegin(), *this); }
        Iterator end() { CheckOk(); return Iterator(m_node->MemberEnd(), *this); }
        ConstIterator begin() const { CheckOk(); return ConstIterator(m_node->MemberBegin(), *this); }
        ConstIterator end() const { CheckOk(); return ConstIterator(m_node->MemberEnd(), *this); }

        template<typename T>
        void AddValue(const std::string& name, const T& value)
        {
            CheckOk();
            m_node->AddMember(rapidjson::Value(name.c_str(), *m_allocator).Move(), value, *m_allocator);
        }

        void AddValue(const std::string& name, const std::string& value)
        {
            CheckOk();
            m_node->AddMember(rapidjson::Value(name.c_str(), *m_allocator).Move(), 
                              rapidjson::Value(value.c_str(), *m_allocator).Move(), *m_allocator);
        }

        template<typename T>
        void AddArray(const std::string& name, const std::vector<T>& values)
        {
            AddArrayImpl(name, values, [&](rapidjson::Value& jsonArray, const T& element) { jsonArray.PushBack(element, *m_allocator); });
        }

        void AddArray(const std::string& name, const std::vector<std::string>& values)
        {
            AddArrayImpl(name, values, [&](rapidjson::Value& jsonArray, const std::string& element)
                {
                    jsonArray.PushBack(rapidjson::Value(element.c_str(), *m_allocator).Move(), *m_allocator);
                });
        }

        const Node AddChildObject(const std::string& name)
        {
            CheckOk();
            m_node->AddMember(rapidjson::Value(name.c_str(), *m_allocator).Move(), rapidjson::Value().Move(), *m_allocator);
            rapidjson::Value& newNode = (*m_node)[name.c_str()];
            newNode.SetObject();
            return Node(&newNode, *this);
        }

        template<typename T> 
        bool GetValue(const std::string& name, T& value, const uint flags) const
        {
            CheckOk();
            const rapidjson::Value* child = GetChildImpl(name, flags);
            if (!child) { return false; }

            AssertMsgFmt(!child->IsObject() && !child->IsArray(), "Value at '%s' is not a scalar.", name.c_str());

            GetValueImpl<std::is_arithmetic<T>::value, std::is_integral<T>::value>::f(*child, name, value);
            return true;
        }

        const Node GetChild(const std::string& name, const uint flags) const
        {
            CheckOk();
            return Node(GetChildImpl(name, flags), *this, name);
        }

        const Node GetChildObject(const std::string& name, const uint flags) const
        {
            CheckOk();
            rapidjson::Value* child = GetChildImpl(name, flags);
            if (!child || !child->IsObject()) { return nullptr; }

            return Node(child, *this, name);
        }

        const Node GetChildArray(const std::string& name, const uint flags) const
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

        template<typename Type>
        bool GetArrayValues(const std::string& name, std::vector<Type>& values, const uint flags) const
        {
            const Node child = GetChildArray(name, flags);
            if (!child) { return false; }
            rapidjson::Value& array = *child.m_node;

            for (size_t idx = 0; idx < array.Size(); idx++)
            {
                Type value;
                GetValueImpl< std::is_arithmetic<Type>::value,
                    std::is_integral<Type>::value >::f(array[idx],
                        "[unknown; getVector()]",
                        value);
                values.emplace_back(value);
            }
            return true;
        }

        template<typename VecType>
        bool GetVector(const std::string& name, VecType& vec, const uint flags) const
        {
            std::vector<typename VecType::kType> values;
            if (!GetArrayValues(name, values, flags)) { return false; }
            AssertMsgFmt(VecType::kDims == values.size(),
                "Error: JSON array '%s' expects %i elements.", VecType::kDims);

            for (int i = 0; i < values.size(); i++) { vec[i] = values[i]; }           
            return true;
        }

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
        }

        void Clear()
        {
            m_document.SetNull();
            m_document.SetObject();
        }        

        void Parse(const std::string& data);
        void Load(const std::string& filePath);
        void DeepCopy(const Document& other);
        std::string Stringify();

    private:
        std::string LoadTextFile(const std::string& filePath) const;
        bool GetFileHandle(const std::string& filePath, std::ifstream& file) const;

    private:
        rapidjson::Document m_document;
    };

    template<> struct Node::GetValueImpl<true, true>
    {
        template<typename T> static void f(const rapidjson::Value& node, const std::string& name, T& value)
        {
            AssertMsgFmt(node.IsInt() || node.IsInt64(), "Node '%s' is not an integer type.", name.c_str());
            value = T(node.GetInt64());
        }

        static void f(const rapidjson::Value& node, const std::string& name, bool& value)
        {
            value = node.GetBool();
        }
    };

    template<> struct Node::GetValueImpl<true, false>
    {
        template<typename T> static void f(const rapidjson::Value& node, const std::string& name, T& value)
        {
            //PT_ASSERT_DETAILED(node.IsDouble()/* || node.IsFloat()*/,
            //                     "Node '%s' is not a floating point type.", name.c_str());
            value = T(node.GetDouble());
        }
    };
}