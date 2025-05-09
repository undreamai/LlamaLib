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

//=================================== HELPERS ===================================//

std::string join_paths(const std::string& a, const std::string& b) {
#ifdef _WIN32
    const char sep = '\\';
#else
    const char sep = '/';
#endif
    if (a.empty()) return b;
    if (b.empty()) return a;
    if (a.back() == sep) return a + b;
    return a + sep + b;
}

const std::vector<std::string> available_architectures(GPU gpu) {
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

LibHandle load_library_safe(const std::string& path) {
    if (setjmp(sigjmp_buf_point) != 0) {
        std::cerr << "Error loading library: " << path << std::endl;
        return false;
    }

    LibHandle handle_out = load_library(path.c_str());
    if (!handle_out) {
        std::cerr << "Failed to load library: " << path << std::endl;
    }

    return handle_out;
}

//=================================== LLMLib ===================================//

LLMLib::~LLMLib() {
    if (llm) {
        LLM_Delete();
        llm = nullptr;
    }
    if (handle) {
        unload_library(handle);
        handle = nullptr;
    }
}

//============================= EXTERNAL API =============================//

const char* Available_Architectures(GPU gpu)
{
    const std::vector<std::string>& llmlibs = available_architectures(gpu);
    static std::string result;

    std::ostringstream oss;
    for (size_t i = 0; i < llmlibs.size(); ++i) {
        if (i != 0) oss << ",";
        oss << llmlibs[i];
    }
    result = oss.str();
    return result.c_str();
}

LLMLib* Load_LLM_Library_From_Path(const std::string& path) {
    LibHandle handle = load_library_safe(path);
    if (!handle) return nullptr;

    LLMLib* llmlib = new LLMLib();
    llmlib->handle = handle;

#define LOAD_SYMBOL(name, ret, ...) \
    llmlib->name##_fn = reinterpret_cast<name##_Fn>(load_symbol(handle, #name)); \
    if (!llmlib->name##_fn) { \
        std::cerr << "Missing symbol: " << #name << std::endl; \
        return false; \
    }

LLMLIB_FUNCTIONS_ALL(LOAD_SYMBOL)
#undef LOAD_SYMBOL

    return llmlib;
}

LLMLib* Load_LLM_Library(GPU gpu, std::string command, const std::string& baseDir) {
    set_error_handlers(true, false);
    const std::vector<std::string>& llmlibArchs = available_architectures(gpu);

    for (const auto& llmlibArch : llmlibArchs) {
        std::cout << "Trying " << llmlibArch << std::endl;
        std::string llmlibPath = join_paths(baseDir, llmlibArch);

        if (setjmp(sigjmp_buf_point) != 0) {
            std::cout << "Error occurred while loading llmlib: " << llmlibPath << ", trying next." << std::endl;
            continue;
        }

        LLMLib* llmlib = Load_LLM_Library_From_Path(llmlibPath);
        if (!llmlib) continue;
        llmlib->llm = llmlib->LLM_Construct(command.c_str());
        std::cout << "Successfully loaded llmlib: " << llmlibPath << std::endl;
        return llmlib;
    }

    return nullptr;
}