#pragma once

#include "generic/Includes.h"

class WindowsSecurityAttributes 
{
public:
	WindowsSecurityAttributes();
	~WindowsSecurityAttributes();
	SECURITY_ATTRIBUTES* operator&();

protected:
	SECURITY_ATTRIBUTES m_winSecurityAttributes;
	PSECURITY_DESCRIPTOR m_winPSecurityDescriptor;
};

