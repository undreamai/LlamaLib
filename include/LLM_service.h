/// @file LLM_service.h
/// @brief LLM service implementation with server capabilities
/// @ingroup llm
/// @details Provides a concrete implementation of LLMProvider with HTTP server
/// functionality, parameter parsing, and integration with llama.cpp backend

#pragma once

#include "LLM.h"

/// @brief Info-level logging macro for LLama library
#define LLAMALIB_INF(...) LOG_TMPL(GGML_LOG_LEVEL_INFO, -1, __VA_ARGS__)

struct common_params;  ///< Forward declaration of llama.cpp parameters structure
struct server_context; ///< Forward declaration of server context structure

/// @brief Concrete implementation of LLMProvider with server capabilities
/// @details This class provides a full-featured LLM service with HTTP server,
/// parameter configuration, and backend integration with llama.cpp
class UNDREAMAI_API LLMServiceImpl : public LLMProvider
{
public:
    /// @brief Default constructor
    /// @details Creates an uninitialized LLMServiceImpl that must be configured before use
    LLMServiceImpl();

    /// @brief Parameterized constructor
    /// @param model_path Path to the model file
    /// @param num_threads Number of CPU threads (-1 for auto-detection)
    /// @param num_GPU_layers Number of layers to offload to GPU
    /// @param num_parallel Number of parallel processing sequences
    /// @param flash_attention Whether to enable flash attention optimization
    /// @param context_size Maximum context length in tokens
    /// @param batch_size Processing batch size
    /// @param embedding_only Whether to run in embedding-only mode
    /// @param lora_paths Vector of paths to LoRA adapter files
    LLMServiceImpl(const std::string &model_path, int num_threads = -1, int num_GPU_layers = 0, int num_parallel = 1, bool flash_attention = false, int context_size = 4096, int batch_size = 2048, bool embedding_only = false, const std::vector<std::string> &lora_paths = {});

    /// @brief Destructor
    ~LLMServiceImpl();

    /// @brief Create LLMServiceImpl from JSON parameters
    /// @param params_json JSON object containing initialization parameters
    /// @return Pointer to newly created LLMServiceImpl instance
    /// @details Factory method for creating instances from structured parameter data
    /// See https://github.com/ggml-org/llama.cpp/tree/master/tools/server#usage for arguments.
    static LLMServiceImpl *from_params(const json &params_json);

    /// @brief Create LLMServiceImpl from command line string
    /// @param command Command line argument string
    /// @return Pointer to newly created LLMServiceImpl instance
    /// @details Factory method for creating instances from command line arguments
    /// See https://github.com/ggml-org/llama.cpp/tree/master/tools/server#usage for arguments.
    static LLMServiceImpl *from_command(const std::string &command);

    /// @brief Create LLMServiceImpl from argc/argv
    /// @param argc Argument count
    /// @param argv Argument vector
    /// @return Pointer to newly created LLMServiceImpl instance
    /// @details Factory method for creating instances from standard main() parameters
    static LLMServiceImpl *from_command(int argc, char **argv);

    /// @brief Convert JSON parameters to command line arguments
    /// @param params_json JSON object with parameters
    /// @return Vector of C-style argument strings
    /// @details Utility function for converting structured parameters to argv format
    static std::vector<char *> jsonToArguments(const json &params_json);

    /// @brief Initialize from argc/argv parameters
    /// @param argc Argument count
    /// @param argv Argument vector
    /// @details Initialize the service with command line style parameters
    void init(int argc, char **argv);

    /// @brief Initialize from parameter string
    /// @param params_string String containing space-separated parameters
    /// @details Initialize the service by parsing a parameter string
    void init(const std::string &params_string);

    /// @brief Initialize from C-style parameter string
    /// @param params_string C-style string containing parameters
    /// @details C-compatible version of string parameter initialization
    void init(const char *params_string);

    //=================================== LLM METHODS START ===================================//
    /// @brief Set debug level (override)
    /// @param debug_level Debug verbosity level
    void debug(int debug_level) override;

    /// @brief Set logging callback (override)
    /// @param callback Function to receive log messages
    void logging_callback(CharArrayFn callback) override;

    /// @brief Get template JSON (override)
    /// @return JSON string with template information
    std::string get_template_json() override;

    /// @brief Set template from JSON (override)
    /// @param data JSON object containing template data
    void set_template_json(const json &data) override;

    /// @brief Apply template to data (override)
    /// @param data JSON object containing messages
    /// @return Formatted string with template applied
    std::string apply_template_json(const json &data) override;

    /// @brief Tokenize input (override)
    /// @param data JSON object containing text to tokenize
    /// @return JSON string with token data
    std::string tokenize_json(const json &data) override;

    /// @brief Detokenize tokens (override)
    /// @param data JSON object containing tokens
    /// @return Detokenized text string
    std::string detokenize_json(const json &data) override;

    /// @brief Generate embeddings (override)
    /// @param data JSON object containing text to embed
    /// @return JSON string with embedding vectors
    std::string embeddings_json(const json &data) override;

    /// @brief Configure LoRA weights (override)
    /// @param data JSON object with LoRA configuration
    /// @return JSON response string
    std::string lora_weight_json(const json &data) override;

    /// @brief List available LoRA adapters (override)
    /// @return JSON string with LoRA adapter list
    std::string lora_list_json() override;

    /// @brief Generate completion (override)
    /// @param data JSON object with prompt and parameters
    /// @param callback Optional streaming callback function
    /// @param callbackWithJSON Whether callback receives JSON format
    /// @return Generated completion text or JSON
    std::string completion_json(const json &data, CharArrayFn callback = nullptr, bool callbackWithJSON = true) override;

    /// @brief Manage processing slots (override)
    /// @param data JSON object with slot operation parameters
    /// @return JSON response string
    std::string slot_json(const json &data) override;

    /// @brief Cancel running request (override)
    /// @param data JSON object with cancellation parameters
    void cancel_json(const json &data) override;

    /// @brief Start HTTP server (override)
    /// @param host Host address to bind (default: "0.0.0.0")
    /// @param port Port number (0 for auto-selection)
    /// @param API_key Optional API key for authentication
    void start_server(const std::string &host = "0.0.0.0", int port = 0, const std::string &API_key = "") override;

    /// @brief Stop HTTP server (override)
    void stop_server() override;

    /// @brief Wait for server thread completion (override)
    void join_server() override;

    /// @brief Start the LLM service (override)
    void start() override;

    /// @brief Stop the LLM service (override)
    void stop() override;

    /// @brief Wait for service thread completion (override)
    void join_service() override;

    /// @brief Configure SSL certificates (override)
    /// @param SSL_cert Path to SSL certificate file
    /// @param SSL_key Path to SSL private key file
    void set_SSL(const std::string &SSL_cert, const std::string &SSL_key) override;

    /// @brief Check service status (override)
    /// @return true if service is running, false otherwise
    bool started() override;

    /// @brief Get embedding vector dimensions (override)
    /// @return Number of dimensions in embedding vectors
    int embedding_size() override;

    /// @brief Get available processing slot (override)
    /// @return Available slot ID or -1 if none available
    int get_available_slot() override;
    //=================================== LLM METHODS END ===================================//

    static std::string debug_implementation() { return "standalone"; }

protected:
    /// @brief Generate embeddings with HTTP response support
    /// @param data JSON object containing embedding request
    /// @param res HTTP response object (for server mode)
    /// @param is_connection_closed Function to check connection status
    /// @return JSON string with embedding data
    /// @details Protected method used internally for server-based embedding generation
    std::string embeddings_json(const json &data, httplib::Response *res, std::function<bool()> is_connection_closed = always_false);

    /// @brief Configure LoRA weights with HTTP response support
    /// @param data JSON object with LoRA configuration
    /// @param res HTTP response object (for server mode)
    /// @return JSON response string
    /// @details Protected method used internally for server-based LoRA configuration
    std::string lora_weight_json(const json &data, httplib::Response *res);

    /// @brief Generate completion with HTTP response support
    /// @param data JSON object with completion request
    /// @param callback Optional streaming callback function
    /// @param callbackWithJSON Whether callback receives JSON format
    /// @param res HTTP response object (for server mode)
    /// @param is_connection_closed Function to check connection status
    /// @param oaicompat OpenAI compatibility mode flag
    /// @return Generated completion text or JSON
    /// @details Protected method used internally for server-based completion generation
    std::string completion_json(const json &data, CharArrayFn callback, bool callbackWithJSON, httplib::Response *res, std::function<bool()> is_connection_closed = always_false, int oaicompat = 0);

    /// @brief Manage slots with HTTP response support
    /// @param data JSON object with slot operation
    /// @param res HTTP response object (for server mode)
    /// @return JSON response string
    /// @details Protected method used internally for server-based slot management
    std::string slot_json(const json &data, httplib::Response *res);

private:
    common_params *params;                ///< Backend parameters structure
    bool llama_backend_has_init;          ///< Whether backend is initialized
    server_context *ctx_server = nullptr; ///< Server context pointer
    std::unique_ptr<httplib::Server> svr; ///< HTTP server instance
    std::string SSL_cert = "";            ///< SSL certificate path
    std::string SSL_key = "";             ///< SSL private key path

    std::mutex start_stop_mutex;                ///< Mutex for start/stop operations
    std::thread service_thread;                 ///< Service worker thread
    std::condition_variable service_stopped_cv; ///< Service stop condition variable
    bool service_stopped = false;               ///< Service stop flag
    std::thread server_thread;                  ///< HTTP server thread
    std::condition_variable server_stopped_cv;  ///< Server stop condition variable
    bool server_stopped = false;                ///< Server stop flag

    /// @brief Split command line string into arguments
    /// @param inputString String containing space-separated arguments
    /// @return Vector of individual argument strings
    /// @details Helper method for parsing command line strings
    std::vector<std::string> splitArguments(const std::string &inputString);

    /// @brief Initialize chat template
    /// @details Sets up the default chat template for the model
    void init_template();

    /// @brief Auto-detect appropriate chat template
    /// @return Detected chat template string
    /// @details Analyzes the model to determine the best chat template format
    const char *detect_chat_template();

    /// @brief Handle streaming completion generation
    /// @param id_tasks Set of task IDs for the completion
    /// @param callback Optional streaming callback function
    /// @param callbackWithJSON Whether callback receives JSON format
    /// @param return_tokens Whether to return token information
    /// @param sink HTTP data sink for streaming responses
    /// @param is_connection_closed Function to check connection status
    /// @return Generated completion string
    /// @details Internal method for handling streaming text generation
    std::string completion_streaming(
        std::unordered_set<int> id_tasks,
        CharArrayFn callback = nullptr,
        bool callbackWithJSON = true,
        bool return_tokens = false,
        httplib::DataSink *sink = nullptr,
        std::function<bool()> is_connection_closed = always_false);

    /// @brief Validate API key middleware
    /// @param req HTTP request object
    /// @param res HTTP response object
    /// @return true if API key is valid, false otherwise
    /// @details HTTP middleware for validating API key authentication
    bool middleware_validate_api_key(const httplib::Request &req, httplib::Response &res);
};

/// @ingroup c_api
/// @{

extern "C"
{
    /// @brief Set registry for LLMServiceImpl (C API)
    /// @param existing_instance Existing registry instance to use
    /// @details Allows injection of custom registry for LLMServiceImpl instances
    UNDREAMAI_API void LLMService_Registry(LLMProviderRegistry *existing_instance);

    /// @brief Construct LLMServiceImpl instance (C API)
    /// @param model_path Path to model file
    /// @param num_threads Number of CPU threads (-1 for auto)
    /// @param num_GPU_layers Number of GPU layers
    /// @param num_parallel Number of parallel sequences
    /// @param flash_attention Whether to use flash attention
    /// @param context_size Maximum context size
    /// @param batch_size Processing batch size
    /// @param embedding_only Whether embedding-only mode
    /// @param lora_count Number of LoRA paths provided
    /// @param lora_paths Array of LoRA file paths
    /// @return Pointer to new LLMServiceImpl instance
    UNDREAMAI_API LLMServiceImpl *LLMService_Construct(const char *model_path, int num_threads = -1, int num_GPU_layers = 0, int num_parallel = 1, bool flash_attention = false, int context_size = 4096, int batch_size = 2048, bool embedding_only = false, int lora_count = 0, const char **lora_paths = nullptr);

    /// @brief Create LLMServiceImpl from command string (C API)
    /// @param params_string Command line parameter string
    /// @return Pointer to new LLMServiceImpl instance
    /// @details See https://github.com/ggml-org/llama.cpp/tree/master/tools/server#usage for arguments.
    UNDREAMAI_API LLMServiceImpl *LLMService_From_Command(const char *params_string);
}

/// @}