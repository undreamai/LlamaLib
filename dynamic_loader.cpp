#include "dynamic_loader.h"

//============================= ERROR HANDLING =============================//

sigjmp_buf sigjmp_buf_point;

void crash_signal_handler(int sig) {
    fail("Severe error occurred", sig);
    siglongjmp(sigjmp_buf_point, 1);
}

void sigint_signal_handler(int sig){}

//============================= LIBRARY LOADING =============================//

inline LibHandle load_library(const char* path) {
    return LOAD_LIB(path);
}

inline void* load_symbol(LibHandle handle, const char* symbol) {
    return GET_SYM(handle, symbol);
}

inline void unload_library(LibHandle handle) {
    CLOSE_LIB(handle);
}

const char* GetPossibleArchitectures(bool gpu) {
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


bool load_library_safe(const std::string& path, LibHandle& handle_out) {
    if (setjmp(sigjmp_buf_point) != 0) {
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


int tryLoadingBackend(const std::vector<std::string>& backends, std::string command, LLMBackend& backend, LibHandle& handle_out, LLM*& llm) {
    set_error_handlers(true, false);

    for (const auto& backendPath : backends) {
        std::cout << "Trying " << backendPath << std::endl;
        if (setjmp(sigjmp_buf_point) != 0) {
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
