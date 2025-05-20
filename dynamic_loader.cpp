#include "dynamic_loader.h"

//=================================== HELPERS ===================================//

#ifdef _WIN32
    const char SEP = '\\';
#else
    const char SEP = '/';
#endif

std::string join_paths(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (b.empty()) return a;
    if (a.back() == SEP) return a + b;
    return a + SEP + b;
}

inline bool file_exists (const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

const std::vector<std::string> available_architectures(bool gpu) {
    std::vector<std::string> architectures;

    const auto add_library = [&](std::string os, std::string arch, std::string prefix, std::string suffix)
    {
        std::string dash_arch = arch;
        if (arch != "") dash_arch = "-" + dash_arch;
        std::string path = prefix + "undreamai_" + os + dash_arch + "." + suffix;
        if (file_exists(path)) {
            architectures.push_back(path);
        } else {
            path = join_paths(os + dash_arch, path);
            if (file_exists(path)) {
                architectures.push_back(path);
            }
        }
    };

    std::string prefix = "";
    std::string suffix = "";
    std::string os = "";

#if defined(_WIN32) || defined(__linux__)
#if defined(_WIN32)
    prefix = "";
    suffix = "dll";
    os = "windows";
#elif defined(__linux__)
    prefix = "lib";
    suffix = "so";
    os = "linux";
#endif
    if (gpu) {
        add_library(os, "cuda-cu12.2.0-full", prefix, suffix);
        add_library(os, "cuda-cu12.2.0", prefix, suffix);
        add_library(os, "hip", prefix, suffix);
        add_library(os, "vulkan", prefix, suffix);
    }
    if (has_avx512()) add_library(os, "avx512", prefix, suffix);
    else if (has_avx2()) add_library(os, "avx2", prefix, suffix);
    else if (has_avx()) add_library(os, "avx", prefix, suffix);
    add_library(os, "noavx", prefix, suffix);
#elif defined(__APPLE__)
#if TARGET_OS_VISION
    add_library("visionos", "", "lib", "a");
#elif TARGET_OS_IOS
    add_library("ios", "", "lib", "a");
#else
    add_library("macos", "acc", "lib", "dylib");
    add_library("macos", "no_acc", "lib", "dylib");
#endif
#elif defined(__ANDROID__)
    add_library("android", "", "lib", "so");
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
        return nullptr;
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

//============================= API =============================//

const char* Available_Architectures(bool gpu)
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

LLMLib* Load_LLM_Library_From_Path(std::string command, const std::string& path) {
    LibHandle handle = load_library_safe(path);
    if (!handle) return nullptr;

    LLMLib* llmlib = new LLMLib();
    llmlib->handle = handle;

#define LOAD_SYMBOL(name, ret, ...) \
    llmlib->name##_fn = reinterpret_cast<name##_Fn>(load_symbol(handle, #name)); \
    if (!llmlib->name##_fn) { \
        std::cerr << "Missing symbol: " << #name << std::endl; \
        return nullptr; \
    }

LLMLIB_FUNCTIONS_ALL(LOAD_SYMBOL)
#undef LOAD_SYMBOL

    llmlib->llm = llmlib->LLM_Construct(command.c_str());

    return llmlib;
}

LLMLib* Load_LLM_Library(std::string command, const std::string& baseDir) {
    bool gpu = false;

    std::istringstream iss(command);
    std::string arg;
    while(iss >> arg) {
        if (arg == "-ngl" || arg == "--gpu-layers" || arg == "--n-gpu-layers") {
            gpu = true;
            break;
        }
    }

    set_error_handlers(true, false);
    const std::vector<std::string>& llmlibArchs = available_architectures(gpu);

    for (const auto& llmlibArch : llmlibArchs) {
        std::cout << "Trying " << llmlibArch << std::endl;
        std::string llmlibPath = join_paths(baseDir, llmlibArch);

        if (setjmp(sigjmp_buf_point) != 0) {
            std::cout << "Error occurred while loading llmlib: " << llmlibPath << ", trying next." << std::endl;
            continue;
        }

        LLMLib* llmlib = Load_LLM_Library_From_Path(command, llmlibPath);
        if (llmlib) {
            std::cout << "Successfully loaded llmlib: " << llmlibPath << std::endl;
            return llmlib;
        }
    }

    return nullptr;
}