#include "LLM_runtime.h"

//============================= LIBRARY LOADING =============================//

const std::string platform_name()
{
#if defined(_WIN32)
    return "win-x64";
#elif defined(__linux__)
    return "linux-x64";
#elif defined(__APPLE__)
#if defined(__x86_64__)
    return "osx-x64";
#else
    return "osx-arm64";
#endif
#else
    std::cerr << "Unknown platform!" << std::endl;
    return "";
#endif
}

const std::vector<std::string> available_architectures(bool gpu)
{
    std::vector<std::string> architectures;
#if defined(_WIN32)
    std::string prefix = "";
#else
    std::string prefix = "lib";
#endif

#if defined(_WIN32)
    std::string suffix = "dll";
#elif defined(__linux__)
    std::string suffix = "so";
#elif defined(__APPLE__)
    std::string suffix = "dylib";
#else
    std::cerr << "Unknown platform!" << std::endl;
    return architectures;
#endif

    const auto add_library = [&](std::string arch)
    {
        std::string platform = platform_name();
        std::string dash_arch = arch;
        if (arch != "")
            dash_arch = "_" + dash_arch;
        std::string path = prefix + "llamalib_" + platform + dash_arch + "." + suffix;
        architectures.push_back(path);
    };

#if defined(_WIN32) || defined(__linux__)
    if (gpu)
    {
        add_library("cublas");
        add_library("tinyblas");
        add_library("hip");
        add_library("vulkan");
    }
    if (has_avx512())
        add_library("avx512");
    else if (has_avx2())
        add_library("avx2");
    else if (has_avx())
        add_library("avx");
    add_library("noavx");
#elif defined(__APPLE__)
    add_library("acc");
    add_library("no-acc");
#endif
    return architectures;
}

std::string get_current_directory()
{
    return std::filesystem::current_path().string();
}

std::string get_executable_directory()
{
#ifdef _WIN32
    char path[MAX_PATH];
    DWORD result = GetModuleFileNameA(nullptr, path, MAX_PATH);
    if (result == 0 || result == MAX_PATH)
    {
        return get_current_directory();
    }
#elif defined(__APPLE__)
    char path[PATH_MAX];
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) != 0)
    {
        return get_current_directory();
    }
#else
    char path[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", path, PATH_MAX);
    if (count == -1)
    {
        return get_current_directory();
    }
    path[count] = '\0';
#endif
    return std::filesystem::path(path).parent_path().string();
}

std::vector<std::string> get_default_library_env_vars()
{
#ifdef _WIN32
    return {"PATH"};
#elif defined(__APPLE__)
    return {"DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "LD_LIBRARY_PATH"};
#else
    return {"LD_LIBRARY_PATH", "LIBRARY_PATH"};
#endif
}

std::vector<std::string> get_env_library_paths(const std::vector<std::string> &env_vars)
{
    std::vector<std::string> paths;

    for (const auto &env_var : env_vars)
    {
        const char *env_value = std::getenv(env_var.c_str());
        if (!env_value)
            continue;

        std::string env_string(env_value);
        if (env_string.empty())
            continue;

        // Split by path separator
#ifdef _WIN32
        const char delimiter = ';';
#else
        const char delimiter = ':';
#endif

        std::stringstream ss(env_string);
        std::string path_str;
        while (std::getline(ss, path_str, delimiter))
        {
            if (!path_str.empty())
            {
                paths.emplace_back(path_str);
            }
        }
    }

    return paths;
}

std::vector<std::string> get_search_directories()
{
    std::vector<std::string> search_paths;
    // Current directory
    search_paths.push_back(get_current_directory());
    // Executable directory
    auto exe_dir = get_executable_directory();
    search_paths.push_back(exe_dir);

    std::string lib_folder_path = (std::filesystem::path("runtimes") / platform_name() / "native").string();

    search_paths.push_back((std::filesystem::path(exe_dir) / lib_folder_path).string());
    search_paths.push_back((std::filesystem::path(exe_dir) / ".." / lib_folder_path).string());

    for (const std::string &lib_folder_name : {"lib", "libs", "runtimes"})
    {
        search_paths.push_back((std::filesystem::path(exe_dir) / lib_folder_path).string());
        search_paths.push_back((std::filesystem::path(exe_dir) / ".." / lib_folder_path).string());
    }
    // Environment variable paths
    auto default_env_vars = get_default_library_env_vars();
    auto env_paths = get_env_library_paths(default_env_vars);
    search_paths.insert(search_paths.end(), env_paths.begin(), env_paths.end());

    std::vector<std::string> return_paths;
    for (const std::string &search_path : search_paths)
    {
        if (std::filesystem::exists(search_path))
            return_paths.push_back(search_path);
    }
    return return_paths;
}

inline LibHandle load_library(const char *path)
{
    return LOAD_LIB(path);
}

inline void *load_symbol(LibHandle handle, const char *symbol)
{
    return GET_SYM(handle, symbol);
}

inline void unload_library(LibHandle handle)
{
    CLOSE_LIB(handle);
}

LibHandle load_library_safe(const std::string &path)
{
    if (setjmp(get_jump_point()) != 0)
    {
        std::cerr << "Error loading library: " << path << std::endl;
        return nullptr;
    }

    LibHandle handle_out = load_library(path.c_str());
    if (!handle_out)
    {
        std::cerr << "Failed to load library: " << path << std::endl;
    }
    return handle_out;
}

bool LLMRuntime::create_LLM_library_backend(const std::string &command, const std::string &llm_lib_filename)
{
    if (setjmp(get_jump_point()) != 0)
    {
        std::cerr << "Error occurred while loading the library" << std::endl;
        return false;
    }
    auto load_sym = [&](auto &fn_ptr, const char *name)
    {
        fn_ptr = reinterpret_cast<std::decay_t<decltype(fn_ptr)>>(load_symbol(handle, name));
        if (!fn_ptr)
        {
            std::cerr << "Failed to load: " << name << std::endl;
        }
    };

    std::vector<std::filesystem::path> full_paths;
    full_paths.push_back(llm_lib_filename);
    for (const std::filesystem::path &search_path : search_paths)
        full_paths.push_back(search_path / llm_lib_filename);

    ensure_error_handlers_initialized();
    std::cout << "Trying " << llm_lib_filename << std::endl;
    for (const std::filesystem::path &full_path : full_paths)
    {
        if (std::filesystem::exists(full_path) && std::filesystem::is_regular_file(full_path))
        {
            handle = load_library_safe(full_path.string());
            if (!handle)
                continue;

#define DECLARE_AND_LOAD(name, ret, ...) \
    load_sym(this->name, #name);         \
    if (!this->name)                     \
        return false;
            LLM_FUNCTIONS_LIST(DECLARE_AND_LOAD)
#undef DECLARE_AND_LOAD

            LLMService_Registry(&LLMProviderRegistry::instance());
            llm = (LLMProvider *)LLMService_From_Command(command.c_str());
            return true;
        }
    }
    return false;
}

bool LLMRuntime::create_LLM_library(const std::string &command)
{
    bool gpu = has_gpu_layers(command);
    for (const auto &llm_lib_filename : available_architectures(gpu))
    {
        bool success = create_LLM_library_backend(command, llm_lib_filename);
        if (success)
        {
            std::cout << "Successfully loaded: " << llm_lib_filename << std::endl;
            return true;
        }
    }
    return false;
}

//============================= LLMRuntime =============================//

LLMRuntime::LLMRuntime()
{
    search_paths = get_search_directories();
}

LLMRuntime::LLMRuntime(const std::string &model_path, int num_threads, int num_GPU_layers, int num_parallel, bool flash_attention, int context_size, int batch_size, bool embedding_only, const std::vector<std::string> &lora_paths)
    : LLMRuntime()
{
    std::string command = LLM::LLM_args_to_command(model_path, num_threads, num_GPU_layers, num_parallel, flash_attention, context_size, batch_size, embedding_only, lora_paths);
    create_LLM_library(command);
}

LLMRuntime *LLMRuntime::from_command(const std::string &command)
{
    LLMRuntime *llmRuntime = new LLMRuntime();
    llmRuntime->create_LLM_library(command);
    return llmRuntime;
}

LLMRuntime *LLMRuntime::from_command(int argc, char **argv)
{
    return from_command(args_to_command(argc, argv));
}

LLMRuntime::~LLMRuntime()
{
    if (llm)
    {
        LLM_Delete(llm);
        llm = nullptr;
    }
    if (handle)
    {
        unload_library(handle);
        handle = nullptr;
    }
}

//============================= API =============================//

const char *Available_Architectures(bool gpu)
{
    const std::vector<std::string> &llmlibs = available_architectures(gpu);
    thread_local static std::string result;

    std::ostringstream oss;
    for (size_t i = 0; i < llmlibs.size(); ++i)
    {
        if (i != 0)
            oss << ",";
        oss << llmlibs[i];
    }
    result = oss.str();
    return result.c_str();
}

LLMRuntime *LLMRuntime_Construct(const char *model_path, int num_threads, int num_GPU_layers, int num_parallel, bool flash_attention, int context_size, int batch_size, bool embedding_only, int lora_count, const char **lora_paths)
{
    std::vector<std::string> lora_paths_vector;
    if (lora_paths != nullptr && lora_count > 0)
    {
        for (int i = 0; i < lora_count; ++i)
        {
            lora_paths_vector.push_back(std::string(lora_paths[i]));
        }
    }
    LLMRuntime *llmRuntime = new LLMRuntime(model_path, num_threads, num_GPU_layers, num_parallel, flash_attention, context_size, batch_size, embedding_only, lora_paths_vector);

    if (llmRuntime->llm == nullptr)
    {
        delete llmRuntime;
        return nullptr;
    }
    return llmRuntime;
}

LLMRuntime *LLMRuntime_From_Command(const char *command)
{
    LLMRuntime *llmRuntime = new LLMRuntime(std::string(command));
    if (llmRuntime->llm == nullptr)
    {
        delete llmRuntime;
        return nullptr;
    }
    return llmRuntime;
}
