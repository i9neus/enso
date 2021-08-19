#include "FilesystemUtils.h"
#include "StringUtils.h"
#include "Assert.h"

#include <filesystem>

namespace fs = std::filesystem;

bool VerifyFile(const std::string& absolutePath)
{
    AssertMsg(!absolutePath.empty(), "The specified file path is empty.");

    // Check to see whether the root path exists
    fs::path fsPath(absolutePath.c_str());
    return fs::exists(fsPath) && fs::is_regular_file(fsPath);
}

bool VerifyDirectory(const std::string& absolutePath)
{
    AssertMsg(!absolutePath.empty(), "The specified directory path is empty.");

    // Check to see whether the root path exists
    fs::path fsPath(absolutePath.c_str());
    return fs::exists(fsPath) && fs::is_directory(fsPath);
}

bool CreateDirectory(const std::string absolutePath)
{
    if(VerifyDirectory(absolutePath)) { return false; }

    // If the directory already exists, don't try to create it
    fs::path fsPath(absolutePath.c_str());
    fs::create_directories(fsPath);

    AssertMsgFmt(fs::exists(fsPath),
                 "The specified path '%s' could not be created.",
                 absolutePath.c_str());

    return true;
}

bool ReplaceExtension(std::string& absolutePath, const std::string& newExtension)
{
    fs::path fsPath(absolutePath.c_str());

    if(!fsPath.has_extension() || !fsPath.has_stem()) { return false; }

    fsPath.replace_extension(fs::path(newExtension.c_str()));
    absolutePath = fsPath.string();

    return true;
}

std::string GetExtension(const std::string& absolutePath)
{
    fs::path fsPath(absolutePath.c_str());
    return Lowercase(fsPath.extension().string());
}

std::string GetFilename(const std::string& absolutePath)
{
    fs::path fsPath(absolutePath.c_str());
    return fsPath.filename().string();
}

bool ReplaceFilename(std::string& absolutePath, const std::string& newFilename)
{
    fs::path fsPath(absolutePath.c_str());

    if (!fsPath.has_filename() || !fsPath.has_stem()) { return false; }

    fsPath.replace_filename(fs::path(newFilename.c_str()));
    absolutePath = fsPath.string();

    return true;
}

std::string GetFileStem(const std::string& path)
{
    fs::path fsPath(path.c_str());
    return fsPath.stem().string();
}

std::string SlashifyPath(const std::string& input)
{
    std::string output = input;
    if(!output.empty() && output.back() != '/') { output += '/'; }
    return output;
}

std::string DeslashifyPath(const std::string& input)
{
    std::string output = input;
    while(!output.empty() && output.back() == '/') {  output.pop_back(); }
    return output;
}

std::string GetModuleDirectory()
{
    // Otherwise, look for the file in the module directory
    HMODULE module = GetModuleHandleA(NULL);
    char modulePath[2048];
    Assert(GetModuleFileNameA(module, modulePath, 2048) < 2048);

    namespace fs = std::filesystem;
    fs::path fsPath(modulePath);
    return fsPath.remove_filename().string();
}

template<typename HandleType>
bool GetFileHandle(const std::string& filePath, HandleType& file, std::ios_base::openmode mode)
{
    // Try loading the file using the verbatim path
    file.open(filePath, mode);
    if (file.good()) { return true; }

    // Get a path to the module directory
    std::string concatPath = GetModuleDirectory();

#ifdef _DEBUG
    // We're in debug mode so assume that the Release directory has the actual file we're looking for
    concatPath += "..\\Release\\";
    Log::Warning("_DEBUG: Modifying local path to '%s'...\n", concatPath + filePath);
#endif

    file.open(concatPath + filePath, mode);
    return file.good();
}

bool GetInputFileHandle(const std::string& filePath, std::ifstream& file)
{
    return GetFileHandle(filePath, file, std::ios::in);
}

bool GetOutputFileHandle(const std::string& filePath, std::ofstream& file)
{
    return GetFileHandle(filePath, file, std::ios::out);
}

std::string ReadTextFile(const std::string& filePath)
{
    std::ifstream file;
    AssertMsgFmt(GetInputFileHandle(filePath, file), "Couldn't open file '%s'", filePath.c_str());

    const int32_t fileSize = file.tellg();
    std::string data;
    data.reserve(fileSize + 1);

    data.assign(std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>());

    file.close();
    return data;
}

void WriteTextFile(const std::string& filePath, const std::string& data)
{
    std::ofstream file;
    AssertMsgFmt(GetOutputFileHandle(filePath, file), "Couldn't open file '%s'", filePath.c_str());

    file.write(data.data(), sizeof(char) * data.size());

    file.close();
}

bool IsAbsolutePath(const std::string& path)
{
    return path.find('\\') != std::string::npos || path.find('\\') != std::string::npos;
}