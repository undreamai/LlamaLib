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

#define BACKEND_FUNCTIONS_WITH_NOLLM_NOARGS(X) \
    X(StopLogging, void) \
    X(StringWrapper_Construct, StringWrapper*) \

#define BACKEND_FUNCTIONS_WITH_NOLLM_ONEARG(X) \
    X(Logging, void, StringWrapper*) \
    X(StringWrapper_Delete, void, StringWrapper*) \
    X(StringWrapper_GetStringSize, int, StringWrapper*) \
    X(LLM_Construct, LLM*, const char*)

#define BACKEND_FUNCTIONS_WITH_NOLLM_FOURARGS(X) \
    X(StringWrapper_GetString, void, StringWrapper*, char*, int, bool)

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
    BACKEND_FUNCTIONS_WITH_NOLLM_NOARGS(X) \
    BACKEND_FUNCTIONS_WITH_NOLLM_ONEARG(X) \
    BACKEND_FUNCTIONS_WITH_NOLLM_FOURARGS(X) \
    BACKEND_FUNCTIONS_WITH_LLM_NOARGS(X) \
    BACKEND_FUNCTIONS_WITH_LLM_ONEARG(X) \
    BACKEND_FUNCTIONS_WITH_LLM_TWOARGS(X)

// Typedefs
#define DECLARE_TYPEDEF(name, ret, ...) typedef ret (*name##_Fn)(__VA_ARGS__);
    BACKEND_FUNCTIONS_ALL(DECLARE_TYPEDEF)
#undef DECLARE_TYPEDEF

struct LLMBackend {
#define DECLARE_FIELD(name, ret, ...) name##_Fn name##_fn;
        BACKEND_FUNCTIONS_ALL(DECLARE_FIELD)
#undef DECLARE_FIELD

    LibHandle handle = nullptr;
    LLM* llm = nullptr;

#define DECLARE_METHOD_NOLLM_NOARGS(name, ret) \
    inline ret name() { return name##_fn(); }

#define DECLARE_METHOD_NOLLM_ONEARG(name, ret, arg1_type) \
    inline ret name(arg1_type arg1) { return name##_fn(arg1); }

#define DECLARE_METHOD_WITH_NOLLM_FOURARGS(name, ret, arg1_type, arg2_type, arg3_type, arg4_type) \
    inline ret name(arg1_type arg1, arg2_type arg2, arg3_type arg3, arg4_type arg4) { return name##_fn(arg1, arg2, arg3, arg4); }

#define DECLARE_METHOD_LLM_NOARGS(name, ret, ...) \
    inline ret name() { return name##_fn(llm); }

#define DECLARE_METHOD_LLM_ONEARG(name, ret, _, arg1_type) \
    inline ret name(arg1_type arg1) { return name##_fn(llm, arg1); }

#define DECLARE_METHOD_LLM_TWOARGS(name, ret, _, arg1_type, arg2_type) \
    inline ret name(arg1_type arg1, arg2_type arg2) { return name##_fn(llm, arg1, arg2); }

BACKEND_FUNCTIONS_WITH_NOLLM_NOARGS(DECLARE_METHOD_NOLLM_NOARGS)
BACKEND_FUNCTIONS_WITH_NOLLM_ONEARG(DECLARE_METHOD_NOLLM_ONEARG)
BACKEND_FUNCTIONS_WITH_NOLLM_FOURARGS(DECLARE_METHOD_WITH_NOLLM_FOURARGS)
BACKEND_FUNCTIONS_WITH_LLM_NOARGS(DECLARE_METHOD_LLM_NOARGS)
BACKEND_FUNCTIONS_WITH_LLM_ONEARG(DECLARE_METHOD_LLM_ONEARG)
BACKEND_FUNCTIONS_WITH_LLM_TWOARGS(DECLARE_METHOD_LLM_TWOARGS)

#undef DECLARE_METHOD_NOLLM_NOARGS
#undef DECLARE_METHOD_NOLLM_ONEARG
#undef DECLARE_METHOD_WITH_NOLLM_FOURARGS
#undef DECLARE_METHOD_LLM_NOARGS
#undef DECLARE_METHOD_LLM_ONEARG
#undef DECLARE_METHOD_LLM_TWOARGS
};

//============================= WRAPPER FUNCTIONS =============================//

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
