#pragma once

#include <string>

namespace Enso
{
    namespace Host 
    {
        template<typename> class Vector;
    }

    namespace ImageIO
    {
        struct Exr
        {
        public:            
            Exr(const std::string& path);
            ~Exr();

            float* operator*() const { return data; }
            operator bool() const { return data != nullptr; }

        public:
            int width;
            int height;

        protected:
            float* data;
        };
    }
}
