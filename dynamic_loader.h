#include "stringwrapper.h"
#include "archchecker.h"
#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>

#include <iostream>
#include <stdexcept>
#include <memory>
#include <setjmp.h>
#include <windows.h>

#include <signal.h>
#ifdef _WIN32
    // Define custom equivalent for Windows (using SEH)
#include <windows.h>
#define sigjmp_buf jmp_buf
#define sigsetjmp(jb, savemask) setjmp(jb)
#define siglongjmp longjmp
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

inline LibHandle load_library(const char* path) {
    return LOAD_LIB(path);
}

inline void* load_symbol(LibHandle handle, const char* symbol) {
    return GET_SYM(handle, symbol);
}

inline void unload_library(LibHandle handle) {
    CLOSE_LIB(handle);
}

extern "C" LOADER_API const char* GetPossibleArchitectures(bool gpu = false) {
    static std::string result;
    std::vector<std::string> architectures;

#if defined(_WIN32) || defined(__linux__)
    if (gpu) {
        architectures.push_back("cuda-cu12.2.0"); // optionally detect FullLlamaLib
        architectures.push_back("hip");
        architectures.push_back("vulkan");
    }

    if (has_avx512()) architectures.push_back("avx512");
    else if (has_avx2()) architectures.push_back("avx2");
    else if (has_avx()) architectures.push_back("avx");
    architectures.push_back("noavx");

#elif defined(__APPLE__)
#if defined(__aarch64__) || defined(__arm64__)
    architectures.push_back("arm64-acc");
    architectures.push_back("arm64-no_acc");
#else
    architectures.push_back("x64-acc");
    architectures.push_back("x64-no_acc");
#endif

#elif defined(__ANDROID__)
    architectures.push_back("android");

#elif defined(__APPLE__) && defined(TARGET_OS_VISION)
    architectures.push_back("visionos");

#else
    architectures.push_back("unknown");
#endif

    std::ostringstream oss;
    for (size_t i = 0; i < architectures.size(); ++i) {
        if (i != 0) oss << ",";
        oss << architectures[i];
    }

    result = oss.str();
    return result.c_str();
}


void unload_llm_backend(LibHandle handle) {
    if (handle) unload_library(handle);
}

jmp_buf point;  // Jump buffer to handle crashes
bool crashed = false;

bool load_library_safe(const std::string& path, LibHandle& handle_out) {
    if (setjmp(point) != 0) {
        std::cerr << "Error loading library: " << path << std::endl;
        return false; // If we jump here, it means there was an error during loading.
    }

    handle_out = load_library(path.c_str());
    if (!handle_out) {
        std::cerr << "Failed to load library: " << path << std::endl;
        return false;
    }

    return true;
}

bool load_llm_backend(const std::string& path, LLMBackend& backend, LibHandle& handle_out) {
    if (!load_library_safe(path, handle_out)) {
        return false;  // If loading fails, return false.
    }

    auto handle = handle_out;

#define LOAD_SYMBOL(name, ret, ...) \
    backend.name = reinterpret_cast<name##_Fn>(load_symbol(handle, #name)); \
    if (!backend.name) return false;

    LLM_FUNCTIONS(LOAD_SYMBOL)  // Make sure to call this macro to load functions

#undef LOAD_SYMBOL

        backend.handle = handle;
    return true;
}

void crash_signal_handler(int sig) {
    std::cerr << "Severe error occurred, signal: " << sig << std::endl;
    longjmp(point, 1); // Jump back to a safe point on error.
}

#ifdef _WIN32
void set_error_handlers() {
    signal(SIGSEGV, crash_signal_handler);
    signal(SIGFPE, crash_signal_handler);
}

#else
static void handle_signal(int sig, siginfo_t* dont_care, void* dont_care_either)
{
    crash_signal_handler(sig);
}

void set_error_handlers() {
    init_status();

    // crash signals
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_NODEFER;
    sa.sa_sigaction = handle_signal;

    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGFPE, &sa, NULL);
}
#endif


int tryLoadingBackend(const std::vector<std::string>& backends, std::string command, LLMBackend& backend, LibHandle& handle_out, LLM*& llm) {
    for (const auto& backendPath : backends) {
        std::cout << "Trying " << backendPath << std::endl;
        if (setjmp(point) != 0) {
            std::cout << "Error occurred while loading backend: " << backendPath << ", trying next." << std::endl;
            continue;  // Continue to try the next backend on error.
        }

        if (!load_llm_backend(backendPath, backend, handle_out)) continue;
        std::cout << "load_llm_backend" << std::endl;
        llm = backend.LLM_Construct(command.c_str());
        std::cout << "Successfully loaded backend: " << backendPath << std::endl;
        return 0;
    }
    return -1;
}