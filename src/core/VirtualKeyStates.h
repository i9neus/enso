#include <stdio.h>

namespace Enso
{
    // Redeclared here so we don't have to include the full contents of WinUser.h
    enum VirtualKeyStates : unsigned int
    {
        VK_LBUTTON      = 0x01,
        VK_RBUTTON      = 0x02,
        VK_MBUTTON      = 0x04,

        VK_BACK         = 0x08,
        VK_TAB          = 0x09,
        VK_CLEAR        = 0x0C,
        VK_RETURN       = 0x0D,
        VK_SHIFT        = 0x10,
        VK_CONTROL      = 0x11,
        VK_MENU         = 0x12,
        VK_PAUSE        = 0x13,
        VK_CAPITAL      = 0x14,

        VK_ESCAPE       = 0x1B
    };
}