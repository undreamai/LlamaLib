#include "dynamic_loader.h"
#include <iostream>
#include <setjmp.h>

//============================= ERROR HANDLING =============================//

sigjmp_buf sigjmp_buf_point;

void crash_signal_handler(int sig) {
    fail("Severe error occurred", sig);
    siglongjmp(sigjmp_buf_point, 1);
}

void sigint_signal_handler(int sig) {}

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

const std::vector<std::string> get_possible_architectures_array(GPU gpu) {
    std::vector<std::string> architectures;
    std::string prefix = "undreamai_";

#if defined(_WIN32) || defined(__linux__)
#if defined(_WIN32)
    prefix += "windows-";
#elif defined(__linux__)
    prefix += "linux-";
#endif
    if (gpu == TINYBLAS) {
        architectures.push_back(prefix + "cuda-cu12.2.0");
    }
    else if (gpu == CUDA) {
        architectures.push_back(prefix + "cuda-cu12.2.0-full");
    }
    if (gpu != NO_GPU) {
        architectures.push_back(prefix + "hip");
        architectures.push_back(prefix + "vulkan");
    }
    if (has_avx512()) architectures.push_back(prefix + "avx512");
    else if (has_avx2()) architectures.push_back(prefix + "avx2");
    else if (has_avx()) architectures.push_back(prefix + "avx");
    architectures.push_back(prefix + "noavx");
#elif defined(__APPLE__)
#if TARGET_OS_VISION
    architectures.push_back(prefix + "visionos");
#elif TARGET_OS_IOS
    architectures.push_back(prefix + "ios");
#else
    architectures.push_back(prefix + "macos-acc");
    architectures.push_back(prefix + "macos-no_acc");
#endif
#elif defined(__ANDROID__)
    architectures.push_back(prefix + "android");
#endif

    return architectures;
}

const char* get_possible_architectures(GPU gpu)
{
    const std::vector<std::string>& backends = get_possible_architectures_array(gpu);
    static std::string result;

    std::ostringstream oss;
    for (size_t i = 0; i < backends.size(); ++i) {
        if (i != 0) oss << ",";
        oss << backends[i];
    }
    result = oss.str();
    return result.c_str();
}

bool load_library_safe(const std::string& path, LibHandle& handle_out) {
    if (setjmp(sigjmp_buf_point) != 0) {
        std::cerr << "Error loading library: " << path << std::endl;
        return false;
    }

    handle_out = load_library(path.c_str());
    if (!handle_out) {
        std::cerr << "Failed to load library: " << path << std::endl;
        return false;
    }

    return true;
}

bool load_llm_backend(const std::string& path, LLMBackend& backend) {
    LibHandle handle;
    if (!load_library_safe(path, handle)) {
        return false;
    }

#define LOAD_SYMBOL(name, ret, ...) \
    backend.name = reinterpret_cast<name##_Fn>(load_symbol(handle, #name)); \
    if (!backend.name) return false;

    BACKEND_FUNCTIONS_ALL(LOAD_SYMBOL)
#undef LOAD_SYMBOL

        backend.handle = handle;
    return true;
}

void unload_llm_backend(LLMBackend& backend) {
    if (backend.llm && backend.LLM_Delete) {
        backend.LLM_Delete(backend.llm);
        backend.llm = nullptr;
    }
    if (backend.handle) {
        unload_library(backend.handle);
        backend.handle = nullptr;
    }
}

int load_backends_fallback(GPU gpu, std::string command, LLMBackend& backend) {
    set_error_handlers(true, false);
    const std::vector<std::string>& backends = get_possible_architectures_array(gpu);

    for (const auto& backendPath : backends) {
        std::cout << "Trying " << backendPath << std::endl;
        if (setjmp(sigjmp_buf_point) != 0) {
            std::cout << "Error occurred while loading backend: " << backendPath << ", trying next." << std::endl;
            continue;
        }

        if (!load_llm_backend(backendPath, backend)) continue;
        backend.llm = backend.LLM_Construct(command.c_str());
        std::cout << "Successfully loaded backend: " << backendPath << std::endl;
        return 0;
    }

    return -1;
}
