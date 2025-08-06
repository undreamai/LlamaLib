/// @file LLM_runtime.h
/// @brief Runtime loading and management of LLM libraries
/// @ingroup llm
/// @details Provides dynamic library loading capabilities for LLM backends,
/// architecture detection, and cross-platform library management

#pragma once

#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <setjmp.h>
#include <type_traits>
#include <algorithm>
#include <cstdlib>

#include "defs.h"
#include "error_handling.h"
#include "LLM.h"

#if defined(_WIN32) || defined(__linux__)
#include "archchecker.h"
#endif

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

// Platform-specific library loading definitions
#if defined(_WIN32)
#include <windows.h>
#include <libloaderapi.h>
using LibHandle = HMODULE;                                 ///< Windows library handle type
#define LOAD_LIB(path) LoadLibraryA(path)                  ///< Load library macro for Windows
#define GET_SYM(handle, name) GetProcAddress(handle, name) ///< Get symbol macro for Windows
#define CLOSE_LIB(handle) FreeLibrary(handle)              ///< Close library macro for Windows
#else
#include <dlfcn.h>
#include <unistd.h>
#include <limits.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

using LibHandle = void *; ///< Unix library handle type
#define LOAD_LIB(path) dlopen(path, RTLD_LAZY)    ///< Load library macro for Unix
#define GET_SYM(handle, name) dlsym(handle, name) ///< Get symbol macro for Unix
#define CLOSE_LIB(handle) dlclose(handle)         ///< Close library macro for Unix
#endif

//=================================== FUNCTION LISTS ===================================//

/// @brief Macro defining the list of dynamically loaded LLM functions
/// @param M Macro to apply to each function signature
/// @details This macro is used to generate function pointer declarations and loading code
#define LLM_FUNCTIONS_LIST(M)                                                                                     \
    M(LLMService_Registry, void, LLMProviderRegistry *)                                                           \
    M(LLMService_Construct, LLMProvider *, const char *, int, int, int, bool, int, int, bool, int, const char **) \
    M(LLMService_From_Command, LLMProvider *, const char *)

/// @brief Runtime loader for LLM libraries
/// @details This class provides dynamic loading of LLM backend libraries,
/// allowing for flexible deployment and architecture-specific optimizations
class UNDREAMAI_API LLMService : public LLMProvider
{
public:
    /// @brief Default constructor
    /// @details Creates an uninitialized runtime that must load a library before use
    LLMService();

    /// @brief Parameterized constructor
    /// @param model_path Path to the model file
    /// @param num_threads Number of CPU threads (-1 for auto-detection)
    /// @param num_GPU_layers Number of layers to offload to GPU
    /// @param num_parallel Number of parallel slots
    /// @param flash_attention Whether to enable flash attention optimization
    /// @param context_size Maximum context length in tokens
    /// @param batch_size Processing batch size
    /// @param embedding_only Whether to run in embedding-only mode
    /// @param lora_paths Vector of paths to LoRA adapter files
    /// @details Creates and initializes a runtime with the specified parameters
    LLMService(const std::string &model_path, int num_threads = -1, int num_GPU_layers = 0, int num_parallel = 1, bool flash_attention = false, int context_size = 4096, int batch_size = 2048, bool embedding_only = false, const std::vector<std::string> &lora_paths = {});

    /// @brief Destructor
    ~LLMService();

    /// @brief Create runtime from command line string
    /// @param command Command line argument string
    /// @return Pointer to newly created LLMService instance
    /// @details Factory method for creating runtime instances from command arguments.
    /// See https://github.com/ggml-org/llama.cpp/tree/master/tools/server#usage for arguments.
    static LLMService *from_command(const std::string &command);

    /// @brief Create runtime from argc/argv
    /// @param argc Argument count
    /// @param argv Argument vector
    /// @return Pointer to newly created LLMService instance
    /// @details Factory method for creating runtime instances from main() parameters
    static LLMService *from_command(int argc, char **argv);

    LibHandle handle = nullptr; ///< Handle to loaded library
    LLMProvider *llm = nullptr; ///< Pointer to loaded LLM provider instance

    /// @brief Loads LLM library dynamically according to underlying achitecture and creates a LLM based on the command
    /// @param command Command string containing model path and parameters
    /// @return true if library loaded successfully, false otherwise
    bool create_LLM_library(const std::string &command);

    //=================================== LLM METHODS START ===================================//
    /// @brief Set debug level (override - delegates to loaded library)
    /// @param debug_level Debug verbosity level
    void debug(int debug_level) override { ((LLMProvider *)llm)->debug(debug_level); }

    /// @brief Set logging callback (override - delegates to loaded library)
    /// @param callback Function to receive log messages
    void logging_callback(CharArrayFn callback) override { ((LLMProvider *)llm)->logging_callback(callback); }

    /// @brief Start HTTP server (override - delegates to loaded library)
    /// @param host Host address (default: "0.0.0.0")
    /// @param port Port number (0 for auto)
    /// @param API_key Optional API key
    void start_server(const std::string &host = "0.0.0.0", int port = 0, const std::string &API_key = "") override { ((LLMProvider *)llm)->start_server(host, port, API_key); }

    /// @brief Stop HTTP server (override - delegates to loaded library)
    void stop_server() override { ((LLMProvider *)llm)->stop_server(); }

    /// @brief Wait for server completion (override - delegates to loaded library)
    void join_server() override { ((LLMProvider *)llm)->join_server(); }

    /// @brief Start service (override - delegates to loaded library)
    void start() override { ((LLMProvider *)llm)->start(); }

    /// @brief Stop service (override - delegates to loaded library)
    void stop() override
    {
        ((LLMProvider *)llm)->stop();
    }

    /// @brief Wait for service completion (override - delegates to loaded library)
    void join_service() override { ((LLMProvider *)llm)->join_service(); }

    /// @brief Set SSL configuration (override - delegates to loaded library)
    /// @param cert SSL certificate path
    /// @param key SSL private key path
    void set_SSL(const std::string &cert, const std::string &key) override { ((LLMProvider *)llm)->set_SSL(cert, key); }

    /// @brief Check service status (override - delegates to loaded library)
    /// @return true if started, false otherwise
    bool started() override { return ((LLMProvider *)llm)->started(); }

    /// @brief Get embedding size (override - delegates to loaded library)
    /// @return Number of embedding dimensions
    int embedding_size() override { return ((LLMProvider *)llm)->embedding_size(); }

    /// @brief Get available slot (override - delegates to loaded library)
    /// @return Available slot ID
    int get_next_available_slot() override { return ((LLMProvider *)llm)->get_next_available_slot(); }

    /// @brief Tokenize input (override - delegates to loaded library)
    /// @param data JSON tokenization request
    /// @return JSON tokenization response
    std::string tokenize_json(const json &data) override { return ((LLMProvider *)llm)->tokenize_json(data); }

    /// @brief Detokenize tokens (override - delegates to loaded library)
    /// @param data JSON detokenization request
    /// @return Detokenized text
    std::string detokenize_json(const json &data) override { return ((LLMProvider *)llm)->detokenize_json(data); }

    /// @brief Generate embeddings (override - delegates to loaded library)
    /// @param data JSON embedding request
    /// @return JSON embedding response
    std::string embeddings_json(const json &data) override { return ((LLMProvider *)llm)->embeddings_json(data); }

    /// @brief Generate completion (override - delegates to loaded library)
    /// @param data JSON completion request
    /// @param callback Optional streaming callback
    /// @param callbackWithJSON Whether callback uses JSON
    /// @return Generated completion
    std::string completion_json(const json &data, CharArrayFn callback = nullptr, bool callbackWithJSON = true) override { return ((LLMProvider *)llm)->completion_json(data, callback, callbackWithJSON); }

    /// @brief Manage slots (override - delegates to loaded library)
    /// @param data JSON slot operation request
    /// @return JSON slot operation response
    std::string slot_json(const json &data) override { return ((LLMProvider *)llm)->slot_json(data); }

    /// @brief Get template (override - delegates to loaded library)
    /// @return JSON template response
    std::string get_template_json() override { return ((LLMProvider *)llm)->get_template_json(); }

    /// @brief Set template (override - delegates to loaded library)
    /// @param data JSON template data
    void set_template_json(const json &data) override { ((LLMProvider *)llm)->set_template_json(data); }

    /// @brief Apply template (override - delegates to loaded library)
    /// @param data JSON template application request
    /// @return Formatted template result
    std::string apply_template_json(const json &data) override { return ((LLMProvider *)llm)->apply_template_json(data); }

    /// @brief Cancel request (override - delegates to loaded library)
    /// @param data JSON cancellation request
    void cancel_json(const json &data) override { ((LLMProvider *)llm)->cancel_json(data); }

    /// @brief Configure LoRA weights (override - delegates to loaded library)
    /// @param data JSON LoRA configuration
    /// @return JSON LoRA response
    std::string lora_weight_json(const json &data) override { return ((LLMProvider *)llm)->lora_weight_json(data); }

    /// @brief List LoRA adapters (override - delegates to loaded library)
    /// @return JSON LoRA list
    std::string lora_list_json() override { return ((LLMProvider *)llm)->lora_list_json(); }

    std::string debug_implementation() override { return "runtime_detection"; }
    //=================================== LLM METHODS END ===================================//

    /// @brief Declare function pointers for dynamically loaded functions
    /// @details Uses the LLM_FUNCTIONS_LIST macro to declare all required function pointers
#define DECLARE_FN(name, ret, ...) \
    ret (*name)(__VA_ARGS__) = nullptr;
    LLM_FUNCTIONS_LIST(DECLARE_FN)
#undef DECLARE_FN

protected:
    std::vector<std::string> search_paths; ///< Library search paths

    /// @brief Load LLM library backend
    /// @param command Command string with parameters
    /// @param llm_lib_filename Specific library filename to load
    /// @return true if library loaded successfully, false otherwise
    /// @details Internal method for loading specific library files
    bool create_LLM_library_backend(const std::string &command, const std::string &llm_lib_filename);
};

/// @brief Get OS-specific library directory
/// @return Path to platform-specific library directory
/// @details Returns the appropriate library directory for the current operating system
const std::string os_library_dir();

/// @brief Get available architectures for the platform
/// @param gpu Whether to include GPU-enabled architectures
/// @return Vector of available architecture strings
/// @details Detects available CPU/GPU architectures for library selection
const std::vector<std::string> available_architectures(bool gpu);

/// @brief Get directory containing the current executable
/// @return Path to executable directory
/// @details Helper function for locating libraries relative to executable
static std::string get_executable_directory();

/// @brief Get current working directory
/// @return Path to current directory
/// @details Helper function for relative path resolution
static std::string get_current_directory();

/// @brief Get library paths from environment variables
/// @param env_vars Vector of environment variable names to check
/// @return Vector of paths found in environment variables
/// @details Extracts library search paths from specified environment variables
static std::vector<std::string> get_env_library_paths(const std::vector<std::string> &env_vars);

/// @brief Get standard library search directories
/// @return Vector of standard search directory paths
/// @details Returns platform-specific standard directories for library searches
static std::vector<std::string> get_search_directories();

/// @brief Get default environment variables for library paths
/// @return Vector of environment variable names to check for library paths
/// @details Returns platform-specific environment variables used for library loading
std::vector<std::string> get_default_library_env_vars();

//=================================== EXTERNAL API ===================================//

/// @ingroup c_api
/// @{

extern "C"
{
    /// @brief Get available architectures (C API)
    /// @param gpu Whether to include GPU architectures
    /// @return JSON string containing available architectures
    UNDREAMAI_API const char *Available_Architectures(bool gpu);
}

/// @}