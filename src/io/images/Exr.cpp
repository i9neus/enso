#include "Exr.h"
#include "io/Log.h"
#include "io/FilesystemUtils.h"

namespace tinyexr
{
    #include "thirdparty/tinyexr/tinyexr.h"
}

namespace Enso
{
    ImageIO::Exr::Exr(const std::string& path) :
        data(nullptr),
        width(0),
        height(0)
    {
        // Sanity checks to make sure the file exists and that it's an EXR
        if (!FileExists(path))
        {
            Log::Error("Error: texture file '%s' does not exist.", path);
            return;
        }
        if (GetExtension(path) != ".exr")
        {
            Log::Error("Error: texture file '%s' is invalid format for ReadEXR()", path);
            return;
        }

        const char* err = nullptr;
        if (tinyexr::LoadEXR(&data, &width, &height, path.c_str(), &err) != TINYEXR_SUCCESS)
        {
            if (err)
            {
                Log::Error("Error: EXR reader returned error '%s'", err);
                tinyexr::FreeEXRErrorMessage(err);
            }
        }
    }  

    ImageIO::Exr::~Exr()
    {
        if (data)
        {
            std::free(data);
        }
    }
}