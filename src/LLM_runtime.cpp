#include "LLM_runtime.h"

//============================= LIBRARY LOADING =============================//

const std::string os_library_dir() {
#if defined(_WIN32)
    return "windows";
#elif defined(__linux__)
    return "linux";
#elif defined(__APPLE__)
#if TARGET_OS_VISION
    return "visionos";
#elif TARGET_OS_IOS
    return "ios";
#else
    return "macos";
#endif
#elif defined(__ANDROID__)
    return "android";
#endif
    return "";
}

const std::vector<std::string> available_architectures(bool gpu) {
    std::vector<std::string> architectures;

    const auto add_library = [&](std::string os, std::string arch, std::string prefix, std::string suffix)
    {
        std::string dash_arch = arch;
        if (arch != "") dash_arch = "_" + dash_arch;
        std::string path = prefix + "llamalib_" + os + dash_arch + "." + suffix;
        architectures.push_back(path);
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
        add_library(os, "cublas", prefix, suffix);
        add_library(os, "tinyblas", prefix, suffix);
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

std::filesystem::path get_executable_directory() {
#ifdef _WIN32
    char path[MAX_PATH];
    DWORD result = GetModuleFileNameA(nullptr, path, MAX_PATH);
    if (result == 0 || result == MAX_PATH) {
        return std::filesystem::current_path();
    }
    return std::filesystem::path(path).parent_path();
#elif defined(__APPLE__)
    char path[PATH_MAX];
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) != 0) {
        return std::filesystem::current_path();
    }
    return std::filesystem::path(path).parent_path();
#else
    char path[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", path, PATH_MAX);
    if (count == -1) {
        return std::filesystem::current_path();
    }
    path[count] = '\0';
    return std::filesystem::path(path).parent_path();
#endif
}

std::filesystem::path get_current_directory() {
    return std::filesystem::current_path();
}

std::vector<std::string> get_default_library_env_vars() {
#ifdef _WIN32
    return {"PATH"};
#elif defined(__APPLE__)
    return {"DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "LD_LIBRARY_PATH"};
#else
    return {"LD_LIBRARY_PATH", "LIBRARY_PATH"};
#endif
}

std::vector<std::filesystem::path> get_env_library_paths(const std::vector<std::string>& env_vars) {
    std::vector<std::filesystem::path> paths;
    
    for (const auto& env_var : env_vars) {
        const char* env_value = std::getenv(env_var.c_str());
        if (!env_value) continue;
        
        std::string env_string(env_value);
        if (env_string.empty()) continue;
        
        // Split by path separator
#ifdef _WIN32
        const char delimiter = ';';
#else
        const char delimiter = ':';
#endif
        
        std::stringstream ss(env_string);
        std::string path_str;
        while (std::getline(ss, path_str, delimiter)) {
            if (!path_str.empty()) {
                paths.emplace_back(path_str);
            }
        }
    }
    
    return paths;
}

std::vector<std::filesystem::path> get_search_directories() {
    std::vector<std::filesystem::path> search_paths;
    // Current directory
    search_paths.push_back(get_current_directory());
    // Executable directory
    search_paths.push_back(get_executable_directory());
    // Common relative paths from executable
    auto exe_dir = get_executable_directory();
    search_paths.push_back(exe_dir / "libs");
    search_paths.push_back(exe_dir / ".." / "libs");
    std::string os_dir = os_library_dir();
    if (os_dir != "")
    {
        search_paths.push_back(exe_dir / "libs" / os_dir);
        search_paths.push_back(exe_dir / ".." / "libs" / os_dir);
    }    
    // Environment variable paths
    auto default_env_vars = get_default_library_env_vars();
    auto env_paths = get_env_library_paths(default_env_vars);
    search_paths.insert(search_paths.end(), env_paths.begin(), env_paths.end());

    std::vector<std::filesystem::path> return_paths;
    for (const std::filesystem::path& search_path : search_paths) {
        if (std::filesystem::exists(search_path)) return_paths.push_back(search_path);
    }
    return return_paths;
}

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
    LibHandle handle_out = load_library(path.c_str());
    if (!handle_out) {
        std::cerr << "Failed to load library: " << path << std::endl;
    }
    return handle_out;
}

bool LLMRuntime::create_LLM_library_backend(const std::string& command, const std::string& path) {
    auto load_sym = [&](auto& fn_ptr, const char* name) {
        fn_ptr = reinterpret_cast<std::decay_t<decltype(fn_ptr)>>(load_symbol(handle, name));
        if (!fn_ptr) {
            std::cerr << "Failed to load: " << name << std::endl;
        }
    };

    std::vector<std::filesystem::path> full_paths;
    full_paths.push_back(path);
    for (const std::filesystem::path& search_path : search_paths) full_paths.push_back(search_path / path);
    
    ensure_error_handlers_initialized();
    std::cout << "Trying " << path << std::endl;
    for (const std::filesystem::path& full_path : full_paths) {
        if (std::filesystem::exists(full_path) && std::filesystem::is_regular_file(full_path)) {
            if (setjmp(get_jump_point()) != 0) {
                std::cout << "Error occurred while loading the library" << std::endl;
                continue;
            }

            handle = load_library_safe(full_path.c_str());
            if (!handle) continue;
            
            #define DECLARE_AND_LOAD(name, ret, ...) \
            load_sym(this->name, #name); \
            if (!this->name) return false;
            LLM_FUNCTIONS_LIST(DECLARE_AND_LOAD)
            #undef DECLARE_AND_LOAD

            llm = (LLMProvider*) LLMService_From_Command(command.c_str());
            return true;
        }
    }
    return false;
}

bool LLMRuntime::create_LLM_library(const std::string& command, const std::string& path) {
    bool gpu = false;

    std::istringstream iss(command);
    std::string arg;
    while(iss >> arg) {
        if (arg == "-ngl" || arg == "--gpu-layers" || arg == "--n-gpu-layers") {
            gpu = true;
            break;
        }
    }

    std::vector<std::string> llmlibPaths;
    if (is_file(path)) {
        llmlibPaths.push_back(path);
    } else {
        for (const auto& arch : available_architectures(gpu)) llmlibPaths.push_back(join_paths(path, arch));
    }

    for (const auto& llmlibPath : llmlibPaths) {
        bool success = create_LLM_library_backend(command, llmlibPath);
        if (success) {
            std::cout << "Successfully loaded: " << llmlibPath << std::endl;
            return true;
        }
    }

    return false;
}

//============================= LLMRuntime =============================//

LLMRuntime::LLMRuntime(const char* model_path, int num_threads, int num_GPU_layers, int num_parallel, bool flash_attention, int context_size, int batch_size, bool embedding_only, int lora_count, const char** lora_paths, const char* path)
: LLMRuntime(LLM::LLM_args_to_command(model_path, num_threads, num_GPU_layers, num_parallel, flash_attention, context_size, batch_size, embedding_only, lora_count, lora_paths), path) { }

LLMRuntime::LLMRuntime(const std::string& command, const std::string& path)
{
    search_paths = get_search_directories();
    create_LLM_library(command, path);
}

LLMRuntime::LLMRuntime(const char* command, const char* path) : LLMRuntime(std::string(command), std::string(path)) { }

LLMRuntime::LLMRuntime(int argc, char ** argv, const char* path) : LLMRuntime(args_to_command(argc, argv), path) { }

LLMRuntime::~LLMRuntime() {
    if (llm) {
        LLM_Delete(llm);
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
    thread_local static std::string result;

    std::ostringstream oss;
    for (size_t i = 0; i < llmlibs.size(); ++i) {
        if (i != 0) oss << ",";
        oss << llmlibs[i];
    }
    result = oss.str();
    return result.c_str();
}

LLMRuntime* LLMRuntime_Construct(const char* model_path, int num_threads, int num_GPU_layers, int num_parallel, bool flash_attention, int context_size, int batch_size, bool embedding_only, int lora_count, const char** lora_paths, const char* path)
{
    return LLMRuntime_From_Command(LLM::LLM_args_to_command(model_path, num_threads, num_GPU_layers, num_parallel, flash_attention, context_size, batch_size, embedding_only, lora_count, lora_paths), path);
}

LLMRuntime* LLMRuntime_From_Command(const char* command, const char* path) {
    LLMRuntime* lib = new LLMRuntime(command, path);
    if(lib->llm == nullptr)
    {
        delete lib;
        return nullptr;
    }
    return lib;
}
