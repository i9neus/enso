#pragma once

#include "thirdparty/rapidjson/document.h"
#include "thirdparty/rapidjson/allocators.h"
#include "thirdparty/rapidjson/error/en.h"
#include "StdIncludes.h"
 
namespace Json
{
    class Node
    {
    private:
        template<bool/* = is_arithmetic<T>*/, bool/* = is_integral<T>*/> struct GetValueImpl
        {
            template<typename T> static void f(const rapidjson::Value& node, const std::string& name, T& value)
            {
                AssertMsgFmt(node.IsString(), "Node '%s' is not of type string.", name.c_str());
                value = node.GetString();
            }
        };       

    public:
        inline void CheckOk() const { AssertMsg(m_node && m_allocator, "Invalid or unitialised JSON node."); }

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

        Node AddChildObject(const std::string& name)
        {
            CheckOk();
            m_node->AddMember(rapidjson::Value(name.c_str(), *m_allocator).Move(), rapidjson::Value().Move(), *m_allocator);
            rapidjson::Value& newNode = (*m_node)[name.c_str()];
            newNode.SetObject();
            return Node(&newNode, m_allocator);
        }

        template<typename T> T GetValue(const std::string& name, bool required = false)
        {
            CheckOk();
            const rapidjson::Value* child = GetChild(parent, name, required);
            if (!child) { return false; }

            AssertMsgFmt(!child->IsObject() && !child->IsArray(), "Value at '%s' is not a scalar.", name.c_str());

            T value;
            GetValueImpl<std::is_arithmetic<T>::value, std::is_integral<T>::value>::f(*child, name, value);
            return value;
        }

        Node GetChild(const std::string& name, bool required = false)
        {
            CheckOk();
            return Node(GetChildImpl(name, required), m_allocator);
        }

        Node GetChildObject(const std::string& name, bool required = false)
        {
            CheckOk();
            rapidjson::Value* child = GetChildImpl(name, required);
            return Node((child && child->IsObject()) ? child : nullptr, m_allocator);
        }

        inline operator bool() const { return m_node; }
        inline bool operator!() const { return !m_node; }

        inline rapidjson::Value* GetPtr() { return m_node; }

    protected:
        Node() : m_node(nullptr) {}
        Node(rapidjson::Value* node, rapidjson::Document::AllocatorType* allocator) : m_node(node), m_allocator(allocator) {}

        rapidjson::Value* GetChildImpl(const std::string& path, bool required);

    protected:
        rapidjson::Value*                    m_node;
        rapidjson::Document::AllocatorType*  m_allocator;
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

        void Parse(const std::string& data);
        void Load(const std::string& filePath);
        std::string Stringify();

    private:
        std::string LoadTextFile(const std::string& filePath) const;

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