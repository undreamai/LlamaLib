#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <setjmp.h>

#include "defs.h"
#include "archchecker.h"
#include "error_handling.h"

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

//=================================== FUNCTION LISTS ===================================//

// Forward declarations
class LLMService;
struct LLMLib;

#define LLMLIB_FUNCTIONS_NOLLM_NOARGS(X) \
    X(StopLogging, void)

#define LLMLIB_FUNCTIONS_NOLLM_ONEARG(X) \
    X(Logging, void, CharArrayFn*) \
    X(LLM_Construct, LLMService*, const char*)

#define LLMLIB_FUNCTIONS_LLM_NOARGS(X) \
    X(LLM_Delete, void, LLMService*) \
    X(LLM_Start, void, LLMService*) \
    X(LLM_Started, bool, LLMService*) \
    X(LLM_Stop, void, LLMService*) \
    X(LLM_StartServer, void, LLMService*) \
    X(LLM_StopServer, void, LLMService*) \
    X(LLM_Join_Server, void, LLMService*) \
    X(LLM_Join_Service, void, LLMService*) \
    X(LLM_Embedding_Size, int, LLMService*) \
    X(LLM_Status_Code, int, LLMService*) \
    X(LLM_Status_Message, const char*, LLMService*) \
    X(LLM_Lora_List, const char*, LLMService*)

#define LLMLIB_FUNCTIONS_LLM_ONEARG(X) \
    X(LLM_Cancel, void, LLMService*, int) \
    X(LLM_Tokenize, const char*, LLMService*, const char*) \
    X(LLM_Detokenize, const char*, LLMService*, const char*) \
    X(LLM_Embeddings, const char*, LLMService*, const char*) \
    X(LLM_Lora_Weight, const char*, LLMService*, const char*) \
    X(LLM_Slot, const char*, LLMService*, const char*)

#define LLMLIB_FUNCTIONS_LLM_TWOARGS(X) \
    X(LLM_SetSSL, void, LLMService*, const char*, const char*) \
    X(LLM_Completion, const char*, LLMService*, const char*, CharArrayFn)

#define LLMLIB_FUNCTIONS_ALL(X) \
    LLMLIB_FUNCTIONS_NOLLM_NOARGS(X) \
    LLMLIB_FUNCTIONS_NOLLM_ONEARG(X) \
    LLMLIB_FUNCTIONS_LLM_NOARGS(X) \
    LLMLIB_FUNCTIONS_LLM_ONEARG(X) \
    LLMLIB_FUNCTIONS_LLM_TWOARGS(X)

// Typedefs
#define DECLARE_TYPEDEF(name, ret, ...) typedef ret (*name##_Fn)(__VA_ARGS__);
LLMLIB_FUNCTIONS_ALL(DECLARE_TYPEDEF)
#undef DECLARE_TYPEDEF

//=================================== LLMLib ===================================//

class UNDREAMAI_API LLMLib {
public:
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
#define DECLARE_METHOD_LLM_NOARGS(name, ret, ...) \
    inline ret name() { return name##_fn(llm); }
#define DECLARE_METHOD_LLM_ONEARG(name, ret, _, arg1_type) \
    inline ret name(arg1_type arg1) { return name##_fn(llm, arg1); }
#define DECLARE_METHOD_LLM_TWOARGS(name, ret, _, arg1_type, arg2_type) \
    inline ret name(arg1_type arg1, arg2_type arg2) { return name##_fn(llm, arg1, arg2); }

LLMLIB_FUNCTIONS_NOLLM_NOARGS(DECLARE_METHOD_NOLLM_NOARGS)
LLMLIB_FUNCTIONS_NOLLM_ONEARG(DECLARE_METHOD_NOLLM_ONEARG)
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

UNDREAMAI_API const char* Available_Architectures(bool gpu);
UNDREAMAI_API LLMLib* Load_LLM_Library_From_Path(std::string command, const std::string& path);
UNDREAMAI_API LLMLib* Load_LLM_Library(std::string command, const std::string& baseDir = "");
UNDREAMAI_API void Free_LLM_Library(LLMLib* llmlib);

#define EXPORT_WRAPPER_NOLLM_NOARGS(name, ret) \
extern "C" inline UNDREAMAI_API ret LLMLib_##name(LLMLib* llmlib) { \
    return llmlib->name(); \
}
#define EXPORT_WRAPPER_NOLLM_ONEARG(name, ret, arg1_type) \
extern "C" inline UNDREAMAI_API ret LLMLib_##name(LLMLib* llmlib, arg1_type arg1) { \
    return llmlib->name(arg1); \
}
#define EXPORT_WRAPPER_LLM_NOARGS(name, ret, _) \
extern "C" inline UNDREAMAI_API ret LLMLib_##name(LLMLib* llmlib) { \
    return llmlib->name(); \
}
#define EXPORT_WRAPPER_LLM_ONEARG(name, ret, _, arg1_type) \
extern "C" inline UNDREAMAI_API ret LLMLib_##name(LLMLib* llmlib, arg1_type arg1) { \
    return llmlib->name(arg1); \
}
#define EXPORT_WRAPPER_LLM_TWOARGS(name, ret, _, arg1_type, arg2_type) \
extern "C" inline UNDREAMAI_API ret LLMLib_##name(LLMLib* llmlib, arg1_type arg1, arg2_type arg2) { \
    return llmlib->name(arg1, arg2); \
}

LLMLIB_FUNCTIONS_NOLLM_NOARGS(EXPORT_WRAPPER_NOLLM_NOARGS)
LLMLIB_FUNCTIONS_NOLLM_ONEARG(EXPORT_WRAPPER_NOLLM_ONEARG)
LLMLIB_FUNCTIONS_LLM_NOARGS(EXPORT_WRAPPER_LLM_NOARGS)
LLMLIB_FUNCTIONS_LLM_ONEARG(EXPORT_WRAPPER_LLM_ONEARG)
LLMLIB_FUNCTIONS_LLM_TWOARGS(EXPORT_WRAPPER_LLM_TWOARGS)

#undef EXPORT_WRAPPER_NOLLM_NOARGS
#undef EXPORT_WRAPPER_NOLLM_ONEARG
#undef EXPORT_WRAPPER_LLM_NOARGS
#undef EXPORT_WRAPPER_LLM_ONEARG
#undef EXPORT_WRAPPER_LLM_TWOARGS
