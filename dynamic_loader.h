#pragma once

#include "stringwrapper.h"
#include "archchecker.h"
#include "error_handling.h"

#include <string>
#include <sstream>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
using LibHandle = HMODULE;
#define LOAD_LIB(path) LoadLibraryA(path)
#define GET_SYM(handle, name) GetProcAddress(handle, name)
#define CLOSE_LIB(handle) FreeLibrary(handle)
#else
#include <dlfcn.h>
using LibHandle = void*;
#define LOAD_LIB(path) dlopen(path, RTLD_LAZY)
#define GET_SYM(handle, name) dlsym(handle, name)
#define CLOSE_LIB(handle) dlclose(handle)
#endif

#ifdef _WIN32
#ifdef UNDREAMAI_EXPORTS
#define LOADER_API __declspec(dllexport)
#else
#define LOADER_API __declspec(dllimport)
#endif
#else
#define LOADER_API
#endif

struct LLM;
struct LLMBackend;

#define LLM_FUNCTIONS(X) \
    X(Logging, void, StringWrapper*) \
    X(StopLogging, void) \
    X(StringWrapper_Construct, StringWrapper*) \
    X(StringWrapper_Delete, void, StringWrapper*) \
    X(StringWrapper_GetStringSize, int, StringWrapper*) \
    X(StringWrapper_GetString, void, StringWrapper*, char*, int, bool) \
    X(LLM_Construct, LLM*, const char*) \
    X(LLM_Delete, void, LLM*) \
    X(LLM_Start, void, LLM*) \
    X(LLM_Started, bool, LLM*) \
    X(LLM_Stop, void, LLM*) \
    X(LLM_StartServer, void, LLM*) \
    X(LLM_StopServer, void, LLM*) \
    X(LLM_SetSSL, void, LLM*, const char*, const char*) \
    X(LLM_Tokenize, void, LLM*, const char*, StringWrapper*) \
    X(LLM_Detokenize, void, LLM*, const char*, StringWrapper*) \
    X(LLM_Embeddings, void, LLM*, const char*, StringWrapper*) \
    X(LLM_Lora_Weight, void, LLM*, const char*, StringWrapper*) \
    X(LLM_Lora_List, void, LLM*, StringWrapper*) \
    X(LLM_Completion, void, LLM*, const char*, StringWrapper*) \
    X(LLM_Slot, void, LLM*, const char*, StringWrapper*) \
    X(LLM_Cancel, void, LLM*, int) \
    X(LLM_Status, int, LLM*, StringWrapper*) \
    X(LLM_Test, int) \
    X(LLM_Embedding_Size, int, LLM*)

// 1. Typedefs
#define DECLARE_TYPEDEF(name, ret, ...) typedef ret (*name##_Fn)(__VA_ARGS__);
LLM_FUNCTIONS(DECLARE_TYPEDEF)
#undef DECLARE_TYPEDEF

// 2. Accessor declarations
#define DECLARE_ACCESSOR(name, ret, ...) \
    extern "C" LOADER_API name##_Fn get_##name(LLMBackend* backend);
LLM_FUNCTIONS(DECLARE_ACCESSOR)
#undef DECLARE_ACCESSOR

// 3. Full struct
struct LLMBackend {
#define DECLARE_FIELD(name, ret, ...) name##_Fn name;
    LLM_FUNCTIONS(DECLARE_FIELD)
#undef DECLARE_FIELD

    void* handle = nullptr;
};

// Loader utilities
LOADER_API const char* GetPossibleArchitectures(bool gpu = false);
bool load_llm_backend(const std::string& path, LLMBackend& backend, LibHandle& handle_out);
int tryLoadingBackend(const std::vector<std::string>& backends, std::string command, LLMBackend& backend, LibHandle& handle_out, LLM*& llm);
void unload_llm_backend(LibHandle handle);