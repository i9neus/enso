#include "JsonUtils.h"
 
namespace Json
{
    Node::Node()
    {
    }

    Document::Document()
    {
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
    }
}