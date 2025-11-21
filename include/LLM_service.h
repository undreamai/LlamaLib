/// @file LLM_service.h
/// @brief LLM service implementation with server capabilities
/// @ingroup llm
/// @details Provides a concrete implementation of LLMProvider with HTTP server
/// functionality, parameter parsing, and integration with llama.cpp backend

#pragma once

#include "LLM.h"
#include "completion_processor.h"

/// @brief Info-level logging macro for LLama library
#define LLAMALIB_INF(...) LOG_TMPL(GGML_LOG_LEVEL_INFO, -1, __VA_ARGS__)

struct common_params;  ///< Forward declaration of llama.cpp parameters structure
struct server_context; ///< Forward declaration of server context structure
struct server_http_context; ///< Forward declaration of server http context structure
struct server_routes;
struct server_http_req;
struct server_http_res;

using server_http_res_ptr = std::unique_ptr<server_http_res>;
using handler_t = std::function<server_http_res_ptr(const server_http_req & req)>;

/// @brief Concrete implementation of LLMProvider with server capabilities
/// @details This class provides a full-featured LLM service with HTTP server,
/// parameter configuration, and backend integration with llama.cpp
class UNDREAMAI_API LLMService : public LLMProvider
{
public:
    /// @brief Default constructor
    /// @details Creates an uninitialized LLMService that must be configured before use
    LLMService();

    /// @brief Parameterized constructor
    /// @param model_path Path to the model file
    /// @param num_threads Number of CPU threads (-1 for auto-detection)
    /// @param num_GPU_layers Number of layers to offload to GPU
    /// @param num_slots Number of parallel processing sequences
    /// @param flash_attention Whether to enable flash attention optimization
    /// @param context_size Maximum context length in tokens
    /// @param batch_size Processing batch size
    /// @param embedding_only Whether to run in embedding-only mode
    /// @param lora_paths Vector of paths to LoRA adapter files
    LLMService(const std::string &model_path, int num_slots = 1, int num_threads = -1, int num_GPU_layers = 0, bool flash_attention = false, int context_size = 4096, int batch_size = 2048, bool embedding_only = false, const std::vector<std::string> &lora_paths = {});

    /// @brief Destructor
    ~LLMService();

    /// @brief Create LLMService from JSON parameters
    /// @param params_json JSON object containing initialization parameters
    /// @return Pointer to newly created LLMService instance
    /// @details Factory method for creating instances from structured parameter data
    /// See https://github.com/ggml-org/llama.cpp/tree/master/tools/server#usage for arguments.
    static LLMService *from_params(const json &params_json);

    /// @brief Create LLMService from command line string
    /// @param command Command line argument string
    /// @return Pointer to newly created LLMService instance
    /// @details Factory method for creating instances from command line arguments
    /// See https://github.com/ggml-org/llama.cpp/tree/master/tools/server#usage for arguments.
    static LLMService *from_command(const std::string &command);

    /// @brief Create LLMService from argc/argv
    /// @param argc Argument count
    /// @param argv Argument vector
    /// @return Pointer to newly created LLMService instance
    /// @details Factory method for creating instances from standard main() parameters
    static LLMService *from_command(int argc, char **argv);

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

    /// @brief Returns the construct command
    std::string get_command() { return command; }

    //=================================== LLM METHODS START ===================================//

    std::string encapsulate_route(const json &body, handler_t route_handler);

    /// @brief Tokenize text
    /// @param query Text string to tokenize
    /// @return Vector of token IDs
    std::vector<int> tokenize(const std::string &query) override;

    /// @brief Convert tokens to text
    /// @param tokens Vector of token IDs to convert
    /// @return Detokenized text string
    std::string detokenize(const std::vector<int32_t> &tokens) override;

    /// @brief Generate embeddings
    /// @param query Text string to embed
    /// @return Vector of embedding values
    std::vector<float> embeddings(const std::string &query) override;

    /// @brief Generate completion (override)
    /// @param data JSON object with prompt and parameters
    /// @param callback Optional streaming callback function
    /// @param callbackWithJSON Whether callback receives JSON format
    /// @return Generated completion text or JSON
    std::string completion_json(const json &data, CharArrayFn callback = nullptr, bool callbackWithJSON = true) override;

    /// @brief Apply template to messages
    /// @param messages JSON array of chat messages
    /// @return Formatted chat string
    std::string apply_template(const json &messages) override;

    /// @brief Configure LoRA weights
    /// @param loras Vector of LoRA adapters with their scales
    /// @return true if configuration was successful, false otherwise
    bool lora_weight(const std::vector<LoraIdScale> &loras) override;

    /// @brief List available LoRA adapters
    /// @return Vector of available LoRA adapters with paths
    std::vector<LoraIdScalePath> lora_list() override;

    /// @brief Cancel running request (override)
    /// @param data JSON object with cancellation parameters
    void cancel(int id_slot) override;

    /// @brief Start the LLM service (override)
    void start() override;

    /// @brief Check service status (override)
    /// @return true if service is running, false otherwise
    bool started() override;

    /// @brief Stop the LLM service (override)
    void stop() override;

    /// @brief Start HTTP server (override)
    /// @param host Host address to bind (default: "0.0.0.0")
    /// @param port Port number (0 for auto-selection)
    /// @param API_key Optional API key for authentication
    void start_server(const std::string &host = "0.0.0.0", int port = -1, const std::string &API_key = "") override;

    /// @brief Stop HTTP server (override)
    void stop_server() override;

    /// @brief Wait for service thread completion (override)
    void join_service() override;

    /// @brief Wait for server thread completion (override)
    void join_server() override;

    /// @brief Configure SSL certificates (override)
    /// @param SSL_cert Path to SSL certificate file
    /// @param SSL_key Path to SSL private key file
    void set_SSL(const std::string &SSL_cert, const std::string &SSL_key) override;

    /// @brief Get embedding vector dimensions (override)
    /// @return Number of dimensions in embedding vectors
    int embedding_size() override;

    /// @brief Get available processing slot (override)
    /// @return Available slot ID or -1 if none available
    int get_next_available_slot() override;

    /// @brief Set debug level (override)
    /// @param debug_level Debug verbosity level
    void debug(int debug_level) override;

    /// @brief Set logging callback (override)
    /// @param callback Function to receive log messages
    void logging_callback(CharArrayFn callback) override;

    std::string debug_implementation() override { return "standalone"; }
    //=================================== LLM METHODS END ===================================//

protected:
    //=================================== LLM METHODS START ===================================//
    /// @brief Perform slot operation
    /// @param id_slot Slot ID to operate on
    /// @param action Action to perform ("save" or "restore")
    /// @param filepath Path for save/load operation
    /// @return Operation result string
    std::string slot(int id_slot, const std::string &action, const std::string &filepath) override;
    //=================================== LLM METHODS END ===================================//

private:
    std::string command = "";             ///< constructor command
    common_params *params;                ///< Backend parameters structure
    bool llama_backend_has_init;          ///< Whether backend is initialized
    server_context *ctx_server = nullptr; ///< Server context pointer
    server_http_context* ctx_http = nullptr;
    server_routes* routes = nullptr;
    std::unique_ptr<httplib::Server> svr; ///< HTTP server instance

    std::mutex start_stop_mutex;                ///< Mutex for start/stop operations
    std::thread service_thread;                 ///< Service worker thread
    std::condition_variable service_stopped_cv; ///< Service stop condition variable
    bool service_stopped = false;               ///< Service stop flag
    std::thread server_thread;                  ///< HTTP server thread
    std::condition_variable server_stopped_cv;  ///< Server stop condition variable
    bool server_stopped = false;                ///< Server stop flag

    int next_available_slot = 0;

    /// @brief Split command line string into arguments
    /// @param inputString String containing space-separated arguments
    /// @return Vector of individual argument strings
    /// @details Helper method for parsing command line strings
    std::vector<std::string> splitArguments(const std::string &inputString);

    /// @brief Auto-detect appropriate chat template
    /// @return Detected chat template string
    /// @details Analyzes the model to determine the best chat template format
    const std::string detect_chat_template();

    /// @brief Tokenize input (override)
    /// @param data JSON object containing text to tokenize
    /// @return JSON string with token data
    std::string tokenize_json(const json &data);

    /// @brief Convert tokens back to text
    /// @param data JSON object containing token IDs
    /// @return JSON string containing detokenized text
    /// @details Pure virtual method for converting token sequences back to text
    std::string detokenize_json(const json &data);

    /// @brief Generate embeddings with HTTP response support
    /// @param data JSON object containing embedding request
    /// @param res HTTP response object (for server mode)
    /// @param is_connection_closed Function to check connection status
    /// @return JSON string with embedding data
    /// @details Protected method used internally for server-based embedding generation
    std::string embeddings_json(const json &data, httplib::Response *res = nullptr, std::function<bool()> is_connection_closed = always_false);

    /// @brief Escape reasoning by adding think tokens
    /// @param server_http_req request with original prompt
    /// @return request with prompt including think tokens
    server_http_req escape_reasoning(server_http_req prompt);

    /// @brief Apply a chat template to message data
    /// @param data JSON object containing messages to format
    /// @return Formatted string with template applied
    /// @details Pure virtual method for applying chat templates to conversation data
    std::string apply_template_json(const json &data);

    /// @brief Configure LoRA weights with HTTP response support
    /// @param data JSON object with LoRA configuration
    /// @param res HTTP response object (for server mode)
    /// @return JSON response string
    /// @details Protected method used internally for server-based LoRA configuration
    std::string lora_weight_json(const json &data, httplib::Response *res = nullptr);

    /// @brief List available LoRA adapters
    /// @return JSON string containing list of available LoRA adapters
    std::string lora_list_json();

    /// @brief Manage slots with HTTP response support
    /// @param data JSON object with slot operation
    /// @param res HTTP response object (for server mode)
    /// @return JSON response string
    /// @details Protected method used internally for server-based slot management
    std::string slot_json(const json &data, httplib::Response *res = nullptr);
};

/// @ingroup c_api
/// @{

extern "C"
{
    /// @brief Set registry for LLMService (C API)
    /// @param existing_instance Existing registry instance to use
    /// @details Allows injection of custom registry for LLMService instances
    UNDREAMAI_API void LLMService_Registry(LLMProviderRegistry *existing_instance);

    /// @brief Construct LLMService instance (C API)
    /// @param model_path Path to model file
    /// @param num_slots Number of parallel sequences
    /// @param num_threads Number of CPU threads (-1 for auto)
    /// @param num_GPU_layers Number of GPU layers
    /// @param flash_attention Whether to use flash attention
    /// @param context_size Maximum context size
    /// @param batch_size Processing batch size
    /// @param embedding_only Whether embedding-only mode
    /// @param lora_count Number of LoRA paths provided
    /// @param lora_paths Array of LoRA file paths
    /// @return Pointer to new LLMService instance
    UNDREAMAI_API LLMService *LLMService_Construct(const char *model_path, int num_slots = 1, int num_threads = -1, int num_GPU_layers = 0, bool flash_attention = false, int context_size = 4096, int batch_size = 2048, bool embedding_only = false, int lora_count = 0, const char **lora_paths = nullptr);

    /// @brief Create LLMService from command string (C API)
    /// @param params_string Command line parameter string
    /// @return Pointer to new LLMService instance
    /// @details See https://github.com/ggml-org/llama.cpp/tree/master/tools/server#usage for arguments.
    UNDREAMAI_API LLMService *LLMService_From_Command(const char *params_string);

    /// @brief Returns the construct command (C API)
    /// @param llm_service the LLMService instance
    UNDREAMAI_API const char *LLMService_Command(LLMService *llm_service);

    UNDREAMAI_API void LLMService_InjectErrorState(ErrorState *error_state);
}

/// @}