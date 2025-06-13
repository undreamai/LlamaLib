#pragma once

#include <string>
#include "json.hpp"
#include <sys/stat.h>

#ifdef _WIN32
#ifdef UNDREAMAI_EXPORTS
#define UNDREAMAI_API __declspec(dllexport)
#else
#define UNDREAMAI_API __declspec(dllimport)
#endif
#else
#define UNDREAMAI_API
#endif

using json = nlohmann::ordered_json;

typedef void (*CharArrayFn)(const char*);

#ifdef _WIN32
    const char SEP = '\\';
#else
    const char SEP = '/';
#endif

inline std::string join_paths(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (b.empty()) return a;
    if (a.back() == SEP) return a + b;
    return a + SEP + b;
}

inline bool file_exists(const std::string& path) {
    struct stat s;
    return stat(path.c_str(), &s) == 0;
}

inline bool is_file(const std::string& path) {
    struct stat s;
    return stat(path.c_str(), &s) == 0 && s.st_mode & S_IFREG;
}

inline bool is_directory(const std::string& path) {
    struct stat s;
    return stat(path.c_str(), &s) == 0 && s.st_mode & S_IFDIR;
}

inline bool always_false()
{
    return false;
}

inline std::string args_to_command(int argc, char** argv)
{
    std::string command = "";
    for (int i = 1; i < argc; ++i) {
        command += argv[i];
        if (i < argc - 1) command += " ";
    }
    return command;
}
