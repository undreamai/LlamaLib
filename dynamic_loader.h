#include "undreamai.h"

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
    X(LLM_SetTemplate, void, LLM*, const char*) \
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


struct LLMBackend;

// 1. Typedefs
#define DECLARE_TYPEDEF(name, ret, ...) typedef ret (*name##_Fn)(__VA_ARGS__);
LLM_FUNCTIONS(DECLARE_TYPEDEF)
#undef DECLARE_TYPEDEF

// 2. Accessor declarations
#define DECLARE_ACCESSOR(name, ret, ...) \
    extern "C" UNDREAMAI_API name##_Fn get_##name(LLMBackend* backend);
LLM_FUNCTIONS(DECLARE_ACCESSOR)
#undef DECLARE_ACCESSOR

// 3. Full struct
struct LLMBackend {
#define DECLARE_FIELD(name, ret, ...) name##_Fn name;
    LLM_FUNCTIONS(DECLARE_FIELD)
#undef DECLARE_FIELD

    void* handle = nullptr;
};



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

inline LibHandle load_library(const char* path) {
    return LOAD_LIB(path);
}

inline void* load_symbol(LibHandle handle, const char* symbol) {
    return GET_SYM(handle, symbol);
}

inline void unload_library(LibHandle handle) {
    CLOSE_LIB(handle);
}



bool load_llm_backend(const std::string& path, LLMBackend& backend, LibHandle& handle_out) {
    handle_out = load_library(path.c_str());
    if (!handle_out) return false;

    auto handle = handle_out;

#define LOAD_SYMBOL(name, ret, ...) \
    backend.name = reinterpret_cast<name##_Fn>(load_symbol(handle, #name)); \
    if (!backend.name) return false;

    LLM_FUNCTIONS(LOAD_SYMBOL)

#undef LOAD_SYMBOL

    backend.handle = handle;
    return true;
}

void unload_llm_backend(LibHandle handle) {
    if (handle) unload_library(handle);
}

