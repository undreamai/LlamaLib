#pragma once

#include "stringwrapper.h"
#include "archchecker.h"
#include "error_handling.h"

#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <setjmp.h>

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

//=================================== FUNCTION LISTS ===================================//

// Forward declarations
class LLMService;
struct LLMLib;

#define LLMLIB_FUNCTIONS_NOLLM_NOARGS(X) \
    X(StopLogging, void) \
    X(StringWrapper_Construct, StringWrapper*) \

#define LLMLIB_FUNCTIONS_NOLLM_ONEARG(X) \
    X(Logging, void, StringWrapper*) \
    X(StringWrapper_Delete, void, StringWrapper*) \
    X(StringWrapper_GetStringSize, int, StringWrapper*) \
    X(LLM_Construct, LLMService*, const char*)

#define LLMLIB_FUNCTIONS_NOLLM_FOURARGS(X) \
    X(StringWrapper_GetString, void, StringWrapper*, char*, int, bool)

#define LLMLIB_FUNCTIONS_LLM_NOARGS(X) \
    X(LLM_Delete, void, LLMService*) \
    X(LLM_Start, void, LLMService*) \
    X(LLM_Started, bool, LLMService*) \
    X(LLM_Stop, void, LLMService*) \
    X(LLM_StartServer, void, LLMService*) \
    X(LLM_StopServer, void, LLMService*) \
    X(LLM_Embedding_Size, int, LLMService*)

#define LLMLIB_FUNCTIONS_LLM_ONEARG(X) \
    X(LLM_Lora_List, void, LLMService*, StringWrapper*) \
    X(LLM_Cancel, void, LLMService*, int) \
    X(LLM_Status, int, LLMService*, StringWrapper*)

#define LLMLIB_FUNCTIONS_LLM_TWOARGS(X) \
    X(LLM_SetSSL, void, LLMService*, const char*, const char*) \
    X(LLM_Tokenize, void, LLMService*, const char*, StringWrapper*) \
    X(LLM_Detokenize, void, LLMService*, const char*, StringWrapper*) \
    X(LLM_Embeddings, void, LLMService*, const char*, StringWrapper*) \
    X(LLM_Lora_Weight, void, LLMService*, const char*, StringWrapper*) \
    X(LLM_Completion, void, LLMService*, const char*, StringWrapper*) \
    X(LLM_Slot, void, LLMService*, const char*, StringWrapper*)

#define LLMLIB_FUNCTIONS_ALL(X) \
    LLMLIB_FUNCTIONS_NOLLM_NOARGS(X) \
    LLMLIB_FUNCTIONS_NOLLM_ONEARG(X) \
    LLMLIB_FUNCTIONS_NOLLM_FOURARGS(X) \
    LLMLIB_FUNCTIONS_LLM_NOARGS(X) \
    LLMLIB_FUNCTIONS_LLM_ONEARG(X) \
    LLMLIB_FUNCTIONS_LLM_TWOARGS(X)

// Typedefs
#define DECLARE_TYPEDEF(name, ret, ...) typedef ret (*name##_Fn)(__VA_ARGS__);
    LLMLIB_FUNCTIONS_ALL(DECLARE_TYPEDEF)
#undef DECLARE_TYPEDEF

//=================================== LLMLib ===================================//

struct LLMLib {
        LLMLib::LLMLib();
        LLMLib::LLMLib(LLMService* llm);
        ~LLMLib();

        LibHandle handle = nullptr;
        LLMService* llm = nullptr;

#define DECLARE_FIELD(name, ret, ...) name##_Fn name##_fn;
        LLMLIB_FUNCTIONS_ALL(DECLARE_FIELD)
#undef DECLARE_FIELD

#define DECLARE_METHOD_NOLLM_NOARGS(name, ret) \
    inline ret name() { return name##_fn(); }

#define DECLARE_METHOD_NOLLM_ONEARG(name, ret, arg1_type) \
    inline ret name(arg1_type arg1) { return name##_fn(arg1); }

#define DECLARE_METHOD_NOLLM_FOURARGS(name, ret, arg1_type, arg2_type, arg3_type, arg4_type) \
    inline ret name(arg1_type arg1, arg2_type arg2, arg3_type arg3, arg4_type arg4) { return name##_fn(arg1, arg2, arg3, arg4); }

#define DECLARE_METHOD_LLM_NOARGS(name, ret, ...) \
    inline ret name() { return name##_fn(llm); }

#define DECLARE_METHOD_LLM_ONEARG(name, ret, _, arg1_type) \
    inline ret name(arg1_type arg1) { return name##_fn(llm, arg1); }

#define DECLARE_METHOD_LLM_TWOARGS(name, ret, _, arg1_type, arg2_type) \
    inline ret name(arg1_type arg1, arg2_type arg2) { return name##_fn(llm, arg1, arg2); }

LLMLIB_FUNCTIONS_NOLLM_NOARGS(DECLARE_METHOD_NOLLM_NOARGS)
LLMLIB_FUNCTIONS_NOLLM_ONEARG(DECLARE_METHOD_NOLLM_ONEARG)
LLMLIB_FUNCTIONS_NOLLM_FOURARGS(DECLARE_METHOD_NOLLM_FOURARGS)
LLMLIB_FUNCTIONS_LLM_NOARGS(DECLARE_METHOD_LLM_NOARGS)
LLMLIB_FUNCTIONS_LLM_ONEARG(DECLARE_METHOD_LLM_ONEARG)
LLMLIB_FUNCTIONS_LLM_TWOARGS(DECLARE_METHOD_LLM_TWOARGS)

#undef DECLARE_METHOD_NOLLM_NOARGS
#undef DECLARE_METHOD_NOLLM_ONEARG
#undef DECLARE_METHOD_NOLLM_FOURARGS
#undef DECLARE_METHOD_LLM_NOARGS
#undef DECLARE_METHOD_LLM_ONEARG
#undef DECLARE_METHOD_LLM_TWOARGS
};

//=================================== HELPERS ===================================//

std::string join_paths(const std::string& a, const std::string& b);
const std::vector<std::string> available_architectures(bool gpu);

//=================================== EXTERNAL API ===================================//

LOADER_API const char* Available_Architectures(bool gpu);
LOADER_API LLMLib* Load_LLM_Library_From_Path(std::string command, const std::string& path);
LOADER_API LLMLib* Load_LLM_Library(std::string command, const std::string& baseDir="");

#define EXPORT_WRAPPER_NOLLM_NOARGS(name, ret) \
extern "C" inline LOADER_API ret LlamaLib_##name(LLMLib* llmlib) { \
    return llmlib->name(); \
}

#define EXPORT_WRAPPER_NOLLM_ONEARG(name, ret, arg1_type) \
extern "C" inline LOADER_API ret LlamaLib_##name(LLMLib* llmlib, arg1_type arg1) { \
    return llmlib->name(arg1); \
}

#define EXPORT_WRAPPER_NOLLM_FOURARGS(name, ret, arg1_type, arg2_type, arg3_type, arg4_type) \
extern "C" inline LOADER_API ret LlamaLib_##name(LLMLib* llmlib, arg1_type arg1, arg2_type arg2, arg3_type arg3, arg4_type arg4) { \
    return llmlib->name(arg1, arg2, arg3, arg4); \
}

#define EXPORT_WRAPPER_LLM_NOARGS(name, ret, _) \
extern "C" inline LOADER_API ret LlamaLib_##name(LLMLib* llmlib) { \
    return llmlib->name(); \
}

#define EXPORT_WRAPPER_LLM_ONEARG(name, ret, _, arg1_type) \
extern "C" inline LOADER_API ret LlamaLib_##name(LLMLib* llmlib, arg1_type arg1) { \
    return llmlib->name(arg1); \
}

#define EXPORT_WRAPPER_LLM_TWOARGS(name, ret, _, arg1_type, arg2_type) \
extern "C" inline LOADER_API ret LlamaLib_##name(LLMLib* llmlib, arg1_type arg1, arg2_type arg2) { \
    return llmlib->name(arg1, arg2); \
}

LLMLIB_FUNCTIONS_NOLLM_NOARGS(EXPORT_WRAPPER_NOLLM_NOARGS)
LLMLIB_FUNCTIONS_NOLLM_ONEARG(EXPORT_WRAPPER_NOLLM_ONEARG)
LLMLIB_FUNCTIONS_NOLLM_FOURARGS(EXPORT_WRAPPER_NOLLM_FOURARGS)
LLMLIB_FUNCTIONS_LLM_NOARGS(EXPORT_WRAPPER_LLM_NOARGS)
LLMLIB_FUNCTIONS_LLM_ONEARG(EXPORT_WRAPPER_LLM_ONEARG)
LLMLIB_FUNCTIONS_LLM_TWOARGS(EXPORT_WRAPPER_LLM_TWOARGS)

#undef EXPORT_WRAPPER_NOLLM_NOARGS
#undef EXPORT_WRAPPER_NOLLM_ONEARG
#undef EXPORT_WRAPPER_NOLLM_FOURARGS
#undef EXPORT_WRAPPER_LLM_NOARGS
#undef EXPORT_WRAPPER_LLM_ONEARG
#undef EXPORT_WRAPPER_LLM_TWOARGS
