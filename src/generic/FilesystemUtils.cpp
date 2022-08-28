#define NOMINMAX
#include <Windows.h>

#include "FilesystemUtils.h"
#include "StringUtils.h"
#include "Assert.h"

#include <filesystem>
#include <fstream>

#include "generic/Log.h"

namespace fs = std::filesystem;

bool FileExists(const std::string& absolutePath)
{
    if (absolutePath.empty()) { return false; }

    // Check to see whether the root path exists
    fs::path fsPath(absolutePath.c_str());
    return fs::exists(fsPath) && fs::is_regular_file(fsPath);
}

bool DirectoryExists(const std::string& absolutePath)
{
    if (absolutePath.empty()) { return false; }

    // Check to see whether the root path exists
    fs::path fsPath(absolutePath.c_str());
    return fs::exists(fsPath) && fs::is_directory(fsPath);
}

bool CreateDirectory(const std::string absolutePath)
{
    if(DirectoryExists(absolutePath)) { return false; }

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
    fs::path fsPath(absolutePath);
    return Lowercase(fsPath.extension().string());
}

std::string GetFilename(const std::string& absolutePath)
{
    fs::path fsPath(absolutePath);
    return fsPath.filename().string();
}

std::string GetParentDirectory(const std::string& absolutePath)
{
    fs::path fsPath(absolutePath);
    return fsPath.has_parent_path() ? fsPath.parent_path().string() : "";
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
bool GetFileHandle(const std::string& filePath, HandleType& file, std::ios_base::openmode mode, std::string* actualPath = nullptr)
{   
    if (fs::path(filePath).is_absolute())
    {
        // Try loading the file using the verbatim path
        file.open(filePath, mode);
        if (file.is_open())
        {
            if (actualPath) { *actualPath = filePath; }
            return true;
        }
    }

    // Get a path to the module directory
    std::string concatPath = GetModuleDirectory();

#ifdef _DEBUG
    // We're in debug mode so assume that the Release directory has the actual file we're looking for
    concatPath += "..\\Release\\";
    Log::Warning("_DEBUG: Modifying local path to '%s'...\n", concatPath + filePath);
#endif

    concatPath += filePath;
    if (actualPath) { *actualPath = concatPath; }

    if (!fs::path(concatPath).is_absolute()) { return false; }

    file.open(concatPath, mode);
    return file.is_open();
}

bool GetInputFileHandle(const std::string& filePath, std::ifstream& file, std::string* actualPath)
{
    return GetFileHandle(filePath, file, std::ios::in, actualPath);
}

bool GetOutputFileHandle(const std::string& filePath, std::ofstream& file, std::string* actualPath)
{
    return GetFileHandle(filePath, file, std::ios::out, actualPath);
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
    return fs::path(path).is_absolute();
}

std::string MakeAbsolutePath(const std::string& parentPath, const std::string& relativePath)
{
    return (fs::path(parentPath) / fs::path(relativePath)).string();
}

int EnumerateDirectoryFiles(const std::string& sourceDirectory, const std::string& extensionFilter, std::vector<std::string>& outputPaths)
{
    namespace fs = std::filesystem;

    if (!fs::is_directory(sourceDirectory)) { return 0; }

    for (auto const& entry : fs::recursive_directory_iterator(sourceDirectory))
    {
        if (!entry.is_regular_file()) { continue; }

        const auto sourcePath = entry.path();

        if (!extensionFilter.empty())
        {
            auto fileExt = sourcePath.extension().string();
            if (fileExt != extensionFilter) { continue; }
        }

        outputPaths.push_back(sourcePath.string());
    }

    return outputPaths.size();
}