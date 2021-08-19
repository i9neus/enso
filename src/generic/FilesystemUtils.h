#pragma once

#include "StdIncludes.h"

// Ensures that the input path has a trailing slash at the end
std::string SlashifyPath(const std::string& input);

// Removes trailing slashes at the end of a path
std::string DeslashifyPath(const std::string& input);

// Check to see whether a file exists
bool VerifyFile(const std::string& absolutePath);

// Check to see whether a directory exists
bool VerifyDirectory(const std::string& absolutePath);

// Create a directory at the specified path
bool CreateDirectory(const std::string absolutePath);

// Replace the existing file extention with a new one
bool ReplaceExtension(std::string& absolutePath, const std::string& newExtension);

// Get the extension for the path
std::string GetExtension(const std::string& path);

// Get the filename for the path
std::string GetFilename(const std::string& path);

// Get the file stem for the path
std::string GetFileStem(const std::string& path);

// Replace with a filename with a new one
bool ReplaceFilename(std::string& absolutePath, const std::string& newFilename);

// Get the directory that the executable is situated in
std::string GetModuleDirectory();

// Load a text file into a string
std::string ReadTextFile(const std::string& filePath);

// Write a string to a text file 
void WriteTextFile(const std::string& filePath, const std::string& data);

// Get a file handle to the specified path
bool GetInputFileHandle(const std::string& filePath, std::ifstream& file);

// Get a file handle to the specified path
bool GetOutputFileHandle(const std::string& filePath, std::ofstream& file);

// Returns true if the path is absolute i.e. it has at least one slash character
inline bool IsAbsolutePath(const std::string& path);