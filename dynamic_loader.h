#pragma once

#include "stringwrapper.h"
#include "archchecker.h"
#include "error_handling.h"

#include <string>
#include <sstream>
#include <vector>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

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

// Forward declarations
struct LLM;
struct LLMBackend;

// Function lists
#define BACKEND_FUNCTIONS(X) \
    X(Logging, void, StringWrapper*) \
    X(StopLogging, void) \
    X(StringWrapper_Construct, StringWrapper*) \
    X(StringWrapper_Delete, void, StringWrapper*) \
    X(StringWrapper_GetStringSize, int, StringWrapper*) \
    X(StringWrapper_GetString, void, StringWrapper*, char*, int, bool) \
    X(LLM_Construct, LLM*, const char*)

#define BACKEND_FUNCTIONS_WITH_LLM_NOARGS(X) \
    X(LLM_Delete, void, LLM*) \
    X(LLM_Start, void, LLM*) \
    X(LLM_Started, bool, LLM*) \
    X(LLM_Stop, void, LLM*) \
    X(LLM_StartServer, void, LLM*) \
    X(LLM_StopServer, void, LLM*) \
    X(LLM_Embedding_Size, int, LLM*)

#define BACKEND_FUNCTIONS_WITH_LLM_ONEARG(X) \
    X(LLM_Lora_List, void, LLM*, StringWrapper*) \
    X(LLM_Cancel, void, LLM*, int) \
    X(LLM_Status, int, LLM*, StringWrapper*)

#define BACKEND_FUNCTIONS_WITH_LLM_TWOARGS(X) \
    X(LLM_SetSSL, void, LLM*, const char*, const char*) \
    X(LLM_Tokenize, void, LLM*, const char*, StringWrapper*) \
    X(LLM_Detokenize, void, LLM*, const char*, StringWrapper*) \
    X(LLM_Embeddings, void, LLM*, const char*, StringWrapper*) \
    X(LLM_Lora_Weight, void, LLM*, const char*, StringWrapper*) \
    X(LLM_Completion, void, LLM*, const char*, StringWrapper*) \
    X(LLM_Slot, void, LLM*, const char*, StringWrapper*)

#define BACKEND_FUNCTIONS_ALL(X) \
    BACKEND_FUNCTIONS(X) \
    BACKEND_FUNCTIONS_WITH_LLM_NOARGS(X) \
    BACKEND_FUNCTIONS_WITH_LLM_ONEARG(X) \
    BACKEND_FUNCTIONS_WITH_LLM_TWOARGS(X)


// Typedefs
#define DECLARE_TYPEDEF(name, ret, ...) typedef ret (*name##_Fn)(__VA_ARGS__);
    BACKEND_FUNCTIONS_ALL(DECLARE_TYPEDEF)
#undef DECLARE_TYPEDEF

// Backend struct
struct LLMBackend {
#define DECLARE_FIELD(name, ret, ...) name##_Fn name;
    BACKEND_FUNCTIONS_ALL(DECLARE_FIELD)
#undef DECLARE_FIELD

        LibHandle handle = nullptr;
    LLM* llm = nullptr;
};


//============================= WRAPPER FUNCTIONS =============================//

#define DECLARE_WRAPPER_NOARGS(name, ret, _) \
    inline ret name(LLMBackend* backend) { \
        return backend->name(backend->llm); \
    }

    BACKEND_FUNCTIONS_WITH_LLM_NOARGS(DECLARE_WRAPPER_NOARGS)
#undef DECLARE_WRAPPER_NOARGS

#define DECLARE_WRAPPER_ONEARG(name, ret, llm_ptr, arg1_type) \
    inline ret name(LLMBackend* backend, arg1_type arg1) { \
        return backend->name(backend->llm, arg1); \
    }

        BACKEND_FUNCTIONS_WITH_LLM_ONEARG(DECLARE_WRAPPER_ONEARG)
#undef DECLARE_WRAPPER_ONEARG

#define DECLARE_WRAPPER_TWOARGS(name, ret, llm_ptr, arg1_type, arg2_type) \
    inline ret name(LLMBackend* backend, arg1_type arg1, arg2_type arg2) { \
        return backend->name(backend->llm, arg1, arg2); \
    }

        BACKEND_FUNCTIONS_WITH_LLM_TWOARGS(DECLARE_WRAPPER_TWOARGS)
#undef DECLARE_WRAPPER_TWOARGS


// GPU Enum
enum GPU {
    NO_GPU = 0,
    TINYBLAS = 1,
    CUDA = 2
};

// Loader functions
const std::vector<std::string> get_possible_architectures_array(GPU gpu);
LOADER_API const char* get_possible_architectures(GPU gpu);
LOADER_API bool load_llm_backend(const std::string& path, LLMBackend& backend);
LOADER_API int load_backends_fallback(GPU gpu, std::string command, LLMBackend& backend);
LOADER_API void unload_llm_backend(LLMBackend& backend);
