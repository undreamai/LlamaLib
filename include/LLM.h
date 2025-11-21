/// @file LLM.h
/// @brief Core LLM functionality interface and base classes
/// @ingroup llm
/// @details This file provides the foundational interface for Large Language Model operations,
/// including completion, tokenization, embeddings, and LoRA (Low-Rank Adaptation) support.

#pragma once

#include "defs.h"
#include "error_handling.h"
// increase max payload length to allow use of larger context size
#define CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH 1048576
// disable Nagle's algorithm
#define CPPHTTPLIB_TCP_NODELAY true
#include "httplib.h"

/// @brief Structure representing a LoRA adapter with ID and scale
/// @details Used for configuring Low-Rank Adaptation layers in language models
struct LoraIdScale
{
    int id;      ///< Unique identifier for the LoRA adapter
    float scale; ///< Scale factor for the LoRA adapter (typically 0.0 to 1.0)

    /// @brief Equality comparison operator
    /// @param other The other LoraIdScale to compare with
    /// @return true if both id and scale are equal, false otherwise
    bool operator==(const LoraIdScale &other) const
    {
        return id == other.id && scale == other.scale;
    }
};

/// @brief Structure representing a LoRA adapter with ID, scale, and file path
/// @details Extended version of LoraIdScale that includes the filesystem path to the adapter
struct LoraIdScalePath
{
    int id;           ///< Unique identifier for the LoRA adapter
    float scale;      ///< Scale factor for the LoRA adapter
    std::string path; ///< Filesystem path to the LoRA adapter file

    /// @brief Equality comparison operator
    /// @param other The other LoraIdScalePath to compare with
    /// @return true if id, scale, and path are all equal, false otherwise
    bool operator==(const LoraIdScalePath &other) const
    {
        return id == other.id && scale == other.scale && path == other.path;
    }
};

/// @brief Ensures error handlers are properly initialized
/// @details This function must be called before using any LLM functionality to set up
/// proper error handling mechanisms
void ensure_error_handlers_initialized();

/// @brief Abstract base class for Large Language Model operations
/// @details Provides the core interface for LLM functionality including text completion,
/// tokenization, embeddings, and template application. This is the base class that all
/// LLM implementations must inherit from.
class UNDREAMAI_API LLM
{
public:
    int32_t n_keep = 0;       ///< Number of tokens to keep from the beginning of the context
    std::string grammar = ""; ///< Grammar specification in GBNF format or JSON schema
    json completion_params;   ///< JSON object containing completion parameters

    /// @brief Virtual destructor
    virtual ~LLM() = default;

    /// @brief Tokenize text
    /// @param query Text string to tokenize
    /// @return Vector of token IDs
    virtual std::vector<int> tokenize(const std::string &query) = 0;

    /// @brief Convert tokens to text
    /// @param tokens Vector of token IDs to convert
    /// @return Detokenized text string
    virtual std::string detokenize(const std::vector<int32_t> &tokens) = 0;

    /// @brief Generate embeddings
    /// @param query Text string to embed
    /// @return Vector of embedding values
    virtual std::vector<float> embeddings(const std::string &query) = 0;

    /// @brief Set completion parameters
    /// @param completion_params_ JSON object containing completion parameters
    /// @details Parameters may include temperature, n_predict, etc.,
    //  See https://github.com/ggml-org/llama.cpp/tree/master/tools/server#post-completion-given-a-prompt-it-returns-the-predicted-completion for the different parameters
    virtual void set_completion_params(json completion_params_) { completion_params = completion_params_; }

    /// @brief Get current completion parameters
    /// @return JSON string of current completion parameters
    virtual std::string get_completion_params() { return completion_params; }

    /// @brief Generate completion
    /// @param prompt Input text prompt
    /// @param callback Optional callback for streaming
    /// @param id_slot Slot ID for the request (-1 for auto)
    /// @param return_response_json Whether to return full JSON response
    /// @return Generated completion text or JSON response
    virtual std::string completion(const std::string &prompt, CharArrayFn callback = nullptr, int id_slot = -1, bool return_response_json = false);

    /// @brief Generate text completion
    /// @param data JSON object containing prompt and parameters
    /// @param callback Optional callback function for streaming responses
    /// @param callbackWithJSON Whether callback receives JSON or plain text
    /// @return JSON string containing generated completion text or JSON response
    /// @details Pure virtual method for text generation with optional streaming
    virtual std::string completion_json(const json &data, CharArrayFn callback, bool callbackWithJSON) = 0;

    /// @brief Set grammar for constrained generation
    /// @param grammar_ Grammar specification in GBNF format or JSON schema
    /// @details See https://github.com/ggml-org/llama.cpp/tree/master/grammars for format details
    virtual void set_grammar(std::string grammar_) { grammar = grammar_; }

    /// @brief Get current grammar specification
    /// @return Current grammar string
    virtual std::string get_grammar() { return grammar; }

    /// @brief Apply template to messages
    /// @param messages JSON array of chat messages
    /// @return Formatted chat string
    virtual std::string apply_template(const json &messages) = 0;

    /// @brief Check if command line arguments specify GPU layers
    /// @param command Command line string to analyze
    /// @return true if GPU layers are specified, false otherwise
    static bool has_gpu_layers(const std::string &command);

    /// @brief Convert LLM parameters to command line arguments
    /// @param model_path Path to the model file
    /// @param num_slots Number of parallel slots to use
    /// @param num_threads Number of CPU threads to use (-1 for auto)
    /// @param num_GPU_layers Number of layers to offload to GPU
    /// @param flash_attention Whether to use flash attention optimization
    /// @param context_size Maximum context length in tokens (default: 4096, 0 = loaded from model)
    /// @param batch_size Batch size for processing
    /// @param embedding_only Whether to run in embedding-only mode
    /// @param lora_paths Vector of paths to LoRA adapter files
    /// @return Command line string with all parameters
    static std::string LLM_args_to_command(const std::string &model_path, int num_slots = 1, int num_threads = -1, int num_GPU_layers = 0, bool flash_attention = false, int context_size = 4096, int batch_size = 2048, bool embedding_only = false, const std::vector<std::string> &lora_paths = {});

protected:
    /// @brief Build JSON for template application
    /// @param messages JSON array of chat messages
    /// @return JSON object ready for apply_template_json
    virtual json build_apply_template_json(const json &messages);

    /// @brief Parse template application result
    /// @param result JSON response from apply_template_json
    /// @return Formatted chat string
    virtual std::string parse_apply_template_json(const json &result);

    /// @brief Build JSON for tokenization
    /// @param query Text string to tokenize
    /// @return JSON object ready for tokenize_json
    virtual json build_tokenize_json(const std::string &query);

    /// @brief Parse tokenization result
    /// @param result JSON response from tokenize_json
    /// @return Vector of token IDs
    virtual std::vector<int> parse_tokenize_json(const json &result);

    /// @brief Build JSON for detokenization
    /// @param tokens Vector of token IDs to convert
    /// @return JSON object ready for detokenize_json
    virtual json build_detokenize_json(const std::vector<int32_t> &tokens);

    /// @brief Parse detokenization result
    /// @param result JSON response from detokenize_json
    /// @return Detokenized text string
    virtual std::string parse_detokenize_json(const json &result);

    /// @brief Build JSON for embeddings generation
    /// @param query Text string to embed
    /// @return JSON object ready for embeddings_json
    virtual json build_embeddings_json(const std::string &query);

    /// @brief Parse embeddings result
    /// @param result JSON response from embeddings_json
    /// @return Vector of embedding values
    virtual std::vector<float> parse_embeddings_json(const json &result);

    /// @brief Build JSON for completion generation
    /// @param prompt Input text prompt
    /// @param id_slot Slot ID for the request (-1 for auto)
    /// @return JSON object ready for completion_json
    virtual json build_completion_json(const std::string &prompt, int id_slot = -1);

    /// @brief Parse completion result
    /// @param result JSON response from completion_json
    /// @return Generated completion text
    virtual std::string parse_completion_json(const json &result);
};

/// @brief Abstract class for local LLM operations with slot management
/// @details Extends the base LLM class with local-specific functionality including
/// slot management for concurrent requests and state persistence
class UNDREAMAI_API LLMLocal : public LLM
{
public:
    /// @brief Get an available processing slot
    /// @return Available slot ID, or -1 if none determined
    virtual int get_next_available_slot() = 0;

    /// @brief Save slot state to file
    /// @param id_slot Slot ID to save
    /// @param filepath Path to save state file
    /// @return Operation result string
    virtual std::string save_slot(int id_slot, const std::string &filepath) { return slot(id_slot, "save", filepath); }

    /// @brief Load slot state from file
    /// @param id_slot Slot ID to restore
    /// @param filepath Path to state file
    /// @return Operation result string
    virtual std::string load_slot(int id_slot, const std::string &filepath) { return slot(id_slot, "restore", filepath); }

    /// @brief Cancel request
    /// @param id_slot Slot ID to cancel
    virtual void cancel(int id_slot) = 0;

protected:
    /// @brief Perform slot operation
    /// @param id_slot Slot ID to operate on
    /// @param action Action to perform ("save" or "restore")
    /// @param filepath Path for save/load operation
    /// @return Operation result string
    virtual std::string slot(int id_slot, const std::string &action, const std::string &filepath) = 0;

    /// @brief Build JSON for slot operations
    /// @param id_slot Slot ID to operate on
    /// @param action Action to perform ("save" or "restore")
    /// @param filepath Path to save/load slot state
    /// @return JSON object ready for slot_json
    virtual json build_slot_json(int id_slot, const std::string &action, const std::string &filepath);

    /// @brief Parse slot operation result
    /// @param result JSON response from slot_json
    /// @return Operation result string
    virtual std::string parse_slot_json(const json &result);
};

/// @brief Abstract class for LLM service providers
/// @details Extends LLMLocal with server functionality, debugging, logging,
/// and advanced features like LoRA management
class UNDREAMAI_API LLMProvider : public LLMLocal
{
public:
    /// @brief Virtual destructor
    virtual ~LLMProvider();

    /// @brief Configure LoRA weights
    /// @param loras Vector of LoRA adapters with their scales
    /// @return true if configuration was successful, false otherwise
    virtual bool lora_weight(const std::vector<LoraIdScale> &loras) = 0;

    /// @brief List available LoRA adapters
    /// @return Vector of available LoRA adapters with paths
    virtual std::vector<LoraIdScalePath> lora_list() = 0;

    /// @brief enable reasoning
    /// @param reasoning whether to enable reasoning
    virtual void enable_reasoning(bool reasoning) { reasoning_enabled = reasoning; }

    /// @brief Set debug level
    /// @param debug_level Debug verbosity level (0 = off, 1 = LlamaLib messages, 2 and higher = llama.cpp messages and more verbose)
    virtual void debug(int debug_level) = 0;

    /// @brief Set logging callback function
    /// @param callback Function to receive log messages
    virtual void logging_callback(CharArrayFn callback) = 0;

    /// @brief Stop logging
    virtual void logging_stop();

    /// @brief Start the LLM service
    virtual void start() = 0;

    /// @brief Check if service is started
    /// @return true if service is running, false otherwise
    virtual bool started() = 0;

    /// @brief Stop the LLM service
    virtual void stop() = 0;

    /// @brief Start HTTP server
    /// @param host Host address to bind to (default: "0.0.0.0")
    /// @param port Port number to bind to (0 for auto-select)
    /// @param API_key Optional API key for authentication
    virtual void start_server(const std::string &host = "0.0.0.0", int port = -1, const std::string &API_key = "") = 0;

    /// @brief Stop HTTP server
    virtual void stop_server() = 0;

    /// @brief Wait for service thread to complete
    virtual void join_service() = 0;

    /// @brief Wait for server thread to complete
    virtual void join_server() = 0;

    /// @brief Configure SSL certificates
    /// @param SSL_cert SSL certificate
    /// @param SSL_key SSL private key
    virtual void set_SSL(const std::string &SSL_cert, const std::string &SSL_key) = 0;

    /// @brief Get embedding vector size
    /// @return Number of dimensions in embedding vectors
    virtual int embedding_size() = 0;

    /// @brief Implementation debugging
    /// @return "standalore" or "runtime_detection" according to the implementation
    virtual std::string debug_implementation() = 0;

protected:
    bool reasoning_enabled = false; ///< Whether reasoning is enabled

    /// @brief Parse LoRA weight configuration result
    /// @param result JSON response from lora_weight_json
    /// @return true if configuration was successful, false otherwise
    virtual bool parse_lora_weight_json(const json &result);

    /// @brief Build JSON for LoRA weight configuration
    /// @param loras Vector of LoRA adapters with their scales
    /// @return JSON object ready for lora_weight_json
    virtual json build_lora_weight_json(const std::vector<LoraIdScale> &loras);

    /// @brief Parse LoRA list result
    /// @param result JSON response from lora_list_json
    /// @return Vector of available LoRA adapters with paths
    virtual std::vector<LoraIdScalePath> parse_lora_list_json(const json &result);
};

/// @brief Registry for managing LLM provider instances
/// @details Singleton pattern implementation for centralized management of
/// LLM provider instances, debugging, and logging configuration
class LLMProviderRegistry
{
public:
    static bool initialised; ///< Whether the registry has been initialized

    /// @brief Inject a custom registry instance
    /// @param instance Custom registry instance to use
    /// @details Allows registry instance injection when using different dynamic libraries
    static void inject_registry(LLMProviderRegistry *instance)
    {
        custom_instance_ = instance;
        initialised = true;
    }

    /// @brief Get the singleton registry instance
    /// @return Reference to the registry instance
    static LLMProviderRegistry &instance()
    {
        if (custom_instance_)
            return *custom_instance_;

        static LLMProviderRegistry registry;
        initialised = true;
        return registry;
    }

    /// @brief Register an LLM provider instance
    /// @param service Provider instance to register
    /// @details Thread-safe registration of provider instances
    void register_instance(LLMProvider *service)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        instances_.push_back(service);
    }

    /// @brief Unregister an LLM provider instance
    /// @param service Provider instance to unregister
    /// @details Thread-safe removal of provider instances
    void unregister_instance(LLMProvider *service)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        instances_.erase(std::remove(instances_.begin(), instances_.end(), service), instances_.end());
    }

    /// @brief Get all registered provider instances
    /// @return Vector of registered provider pointers
    /// @details Thread-safe access to registered instances
    std::vector<LLMProvider *> get_instances()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return instances_;
    }

    /// @brief Set global debug level
    /// @param level Debug verbosity level
    void set_debug_level(int level)
    {
        debug_level_ = level;
    }

    /// @brief Get current debug level
    /// @return Current debug level
    const int get_debug_level()
    {
        return debug_level_;
    }

    /// @brief Set global log callback
    /// @param callback Function to receive log messages
    void set_log_callback(CharArrayFn callback)
    {
        log_callback_ = callback;
    }

    /// @brief Get current log callback
    /// @return Current log callback function
    const CharArrayFn get_log_callback()
    {
        return log_callback_;
    }

private:
    static LLMProviderRegistry *custom_instance_; ///< Custom injected instance

    std::mutex mutex_;                     ///< Thread synchronization mutex
    std::vector<LLMProvider *> instances_; ///< Registered provider instances
    int debug_level_ = 0;                  ///< Global debug level
    CharArrayFn log_callback_ = nullptr;   ///< Global log callback

    /// @brief Private constructor (singleton pattern)
    LLMProviderRegistry() = default;
    /// @brief Private destructor (singleton pattern)
    ~LLMProviderRegistry() = default;
    /// @brief Deleted copy constructor (singleton pattern)
    LLMProviderRegistry(const LLMProviderRegistry &) = delete;
    /// @brief Deleted assignment operator (singleton pattern)
    LLMProviderRegistry &operator=(const LLMProviderRegistry &) = delete;
};

/// @ingroup c_api
/// @{

extern "C"
{
    /// @brief Check if command has GPU layers (C API)
    /// @param command Command string to check
    /// @return true if GPU layers are specified, false otherwise
    UNDREAMAI_API bool Has_GPU_Layers(const char *command);

    /// @brief Set global debug level (C API)
    /// @param debug_level Debug verbosity level
    UNDREAMAI_API void LLM_Debug(int debug_level);

    /// @brief Set global logging callback (C API)
    /// @param callback Function to receive log messages
    UNDREAMAI_API void LLM_Logging_Callback(CharArrayFn callback);

    /// @brief Stop global logging (C API)
    UNDREAMAI_API void LLM_Logging_Stop();

#ifdef _DEBUG
    /// @brief Check if debugger is attached (C API, debug builds only)
    /// @return true if debugger is attached, false otherwise
    UNDREAMAI_API const bool IsDebuggerAttached(void);
#endif

    /// @brief Set completion parameters (C API)
    /// @param llm LLM instance pointer
    /// @param params_json JSON string with parameters (default: "{}")
    UNDREAMAI_API void LLM_Set_Completion_Parameters(LLM *llm, const char *params_json = "{}");

    /// @brief Get completion parameters (C API)
    /// @param llm LLM instance pointer
    /// @return JSON string with current parameters
    UNDREAMAI_API const char *LLM_Get_Completion_Parameters(LLM *llm);

    /// @brief Set grammar (C API)
    /// @param llm LLM instance pointer
    /// @param grammar Grammar string (default: "")
    UNDREAMAI_API void LLM_Set_Grammar(LLM *llm, const char *grammar = "");

    /// @brief Get grammar (C API)
    /// @param llm LLM instance pointer
    /// @return Current grammar string
    UNDREAMAI_API const char *LLM_Get_Grammar(LLM *llm);

    /// @brief Get chat template (C API)
    /// @param llm LLM instance pointer
    /// @return Chat template string
    UNDREAMAI_API const char *LLM_Get_Template(LLM *llm);

    /// @brief Apply chat template (C API)
    /// @param llm LLM instance pointer
    /// @param messages_as_json JSON string with messages
    /// @return Formatted chat string
    UNDREAMAI_API const char *LLM_Apply_Template(LLM *llm, const char *messages_as_json);

    /// @brief Tokenize text (C API)
    /// @param llm LLM instance pointer
    /// @param query Text to tokenize
    /// @return JSON string with token IDs
    UNDREAMAI_API const char *LLM_Tokenize(LLM *llm, const char *query);

    /// @brief Detokenize tokens (C API)
    /// @param llm LLM instance pointer
    /// @param tokens_as_json JSON string with token IDs
    /// @return Detokenized text
    UNDREAMAI_API const char *LLM_Detokenize(LLM *llm, const char *tokens_as_json);

    /// @brief Generate embeddings (C API)
    /// @param llm LLM instance pointer
    /// @param query Text to embed
    /// @return JSON string with embeddings
    UNDREAMAI_API const char *LLM_Embeddings(LLM *llm, const char *query);

    /// @brief Generate completion (C API)
    /// @param llm LLM instance pointer
    /// @param prompt Input prompt
    /// @param callback Optional streaming callback
    /// @param id_slot Slot ID (-1 for auto)
    /// @param return_response_json Whether to return JSON response
    /// @return Generated text or JSON response
    UNDREAMAI_API const char *LLM_Completion(LLM *llm, const char *prompt, CharArrayFn callback = nullptr, int id_slot = -1, bool return_response_json = false);

    /// @brief Save slot state (C API)
    /// @param llm LLMLocal instance pointer
    /// @param id_slot Slot ID to save
    /// @param filepath Path to save file
    /// @return Operation result string
    UNDREAMAI_API const char *LLM_Save_Slot(LLMLocal *llm, int id_slot, const char *filepath);

    /// @brief Load slot state (C API)
    /// @param llm LLMLocal instance pointer
    /// @param id_slot Slot ID to restore
    /// @param filepath Path to load file
    /// @return Operation result string
    UNDREAMAI_API const char *LLM_Load_Slot(LLMLocal *llm, int id_slot, const char *filepath);

    /// @brief Cancel request (C API)
    /// @param llm LLMLocal instance pointer
    /// @param id_slot Slot ID to cancel
    UNDREAMAI_API void LLM_Cancel(LLMLocal *llm, int id_slot);

    /// @brief Set chat template (C API)
    /// @param llm LLMProvider instance pointer
    /// @param chat_template Template string
    UNDREAMAI_API void LLM_Set_Template(LLMProvider *llm, const char *chat_template);

    /// @brief Configure LoRA weights (C API)
    /// @param llm LLMProvider instance pointer
    /// @param loras_as_json JSON string with LoRA configuration
    /// @return true if successful, false otherwise
    UNDREAMAI_API bool LLM_Lora_Weight(LLMProvider *llm, const char *loras_as_json);

    /// @brief Enable reasoning (C API)
    /// @param llm LLMProvider instance pointer
    /// @param enable_reasoning bool whether to enable reasoning
    UNDREAMAI_API void LLM_Enable_Reasoning(LLMProvider *llm, bool enable_reasoning);

    /// @brief List LoRA adapters (C API)
    /// @param llm LLMProvider instance pointer
    /// @return JSON string with LoRA list
    UNDREAMAI_API const char *LLM_Lora_List(LLMProvider *llm);

    /// @brief Delete LLM provider (C API)
    /// @param llm LLMProvider instance pointer
    UNDREAMAI_API void LLM_Delete(LLMProvider *llm);

    /// @brief Start LLM service (C API)
    /// @param llm LLMProvider instance pointer
    UNDREAMAI_API void LLM_Start(LLMProvider *llm);

    /// @brief Check if service is started (C API)
    /// @param llm LLMProvider instance pointer
    /// @return true if started, false otherwise
    UNDREAMAI_API const bool LLM_Started(LLMProvider *llm);

    /// @brief Stop LLM service (C API)
    /// @param llm LLMProvider instance pointer
    UNDREAMAI_API void LLM_Stop(LLMProvider *llm);

    /// @brief Start HTTP server (C API)
    /// @param llm LLMProvider instance pointer
    /// @param host Host address (default: "0.0.0.0")
    /// @param port Port number (0 for auto)
    /// @param API_key Optional API key
    UNDREAMAI_API void LLM_Start_Server(LLMProvider *llm, const char *host = "0.0.0.0", int port = -1, const char *API_key = "");

    /// @brief Stop HTTP server (C API)
    /// @param llm LLMProvider instance pointer
    UNDREAMAI_API void LLM_Stop_Server(LLMProvider *llm);

    /// @brief Wait for service to complete (C API)
    /// @param llm LLMProvider instance pointer
    UNDREAMAI_API void LLM_Join_Service(LLMProvider *llm);

    /// @brief Wait for server to complete (C API)
    /// @param llm LLMProvider instance pointer
    UNDREAMAI_API void LLM_Join_Server(LLMProvider *llm);

    /// @brief Set SSL configuration (C API)
    /// @param llm LLMProvider instance pointer
    /// @param SSL_cert Path to certificate file
    /// @param SSL_key Path to private key file
    UNDREAMAI_API void LLM_Set_SSL(LLMProvider *llm, const char *SSL_cert, const char *SSL_key);

    /// @brief Get last operation status code (C API)
    /// @return Status code of last operation
    UNDREAMAI_API const int LLM_Status_Code();

    /// @brief Get last operation status message (C API)
    /// @return Status message of last operation
    UNDREAMAI_API const char *LLM_Status_Message();

    /// @brief Get embedding vector size (C API)
    /// @param llm LLMProvider instance pointer
    /// @return Number of dimensions in embeddings
    UNDREAMAI_API const int LLM_Embedding_Size(LLMProvider *llm);
}

/// @}