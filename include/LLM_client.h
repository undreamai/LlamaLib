/// @file LLM_client.h
/// @brief Client interface for local and remote LLM access
/// @ingroup llm
/// @details Provides a unified interface for accessing LLM functionality both
/// locally (in-process) and remotely (via HTTP), with support for streaming responses

#pragma once

#include <vector>
#include <thread>
#include <chrono>
#include <iostream>
#include <sstream>
#include "LLM.h"

/// @brief Context structure for streaming operations
/// @details Used internally to manage streaming response data and callbacks
struct StreamingContext
{
    std::string buffer;             ///< Buffer for accumulating streaming data
    CharArrayFn callback = nullptr; ///< Callback function for processing chunks
};

/// @brief Client for accessing LLM functionality locally or remotely
/// @details Provides a unified interface that can connect to either local LLMProvider
/// instances or remote LLM services via HTTP. Supports all standard LLM operations
/// including completion, tokenization, embeddings, and slot management.
class UNDREAMAI_API LLMClient : public LLMLocal
{
public:
    /// @brief Constructor for local LLM access
    /// @param llm Pointer to local LLMProvider instance
    /// @details Creates a client that directly accesses a local LLM provider
    LLMClient(LLMProvider *llm);

    /// @brief Constructor for remote LLM access
    /// @param url Server URL or hostname
    /// @param port Server port number
    /// @param API_key Optional API key
    /// @details Creates a client that connects to a remote LLM server via HTTP
    LLMClient(const std::string &url, const int port, const std::string &API_key = "");

    /// @brief Destructor
    ~LLMClient();

    /// @brief Configure SSL certificate for remote connections
    /// @param SSL_cert Path to SSL certificate file
    /// @details Only applicable for remote clients. Sets up SSL verification.
    void set_SSL(const char *SSL_cert);

    /// @brief Check if this is a remote client
    /// @return true if configured for remote access, false for local access
    /// @details Helper method to determine the client's connection type
    bool is_remote() const { return !url.empty() && port > -1; }

    //=================================== LLM METHODS START ===================================//
    /// @brief Get template JSON (override)
    /// @return JSON string with template information
    /// @details Forwards request to appropriate backend (local or remote)
    std::string get_template_json() override;

    /// @brief Apply template to messages (override)
    /// @param data JSON object containing messages to format
    /// @return Formatted string with template applied
    /// @details Forwards template application to appropriate backend
    std::string apply_template_json(const json &data) override;

    /// @brief Tokenize text input (override)
    /// @param data JSON object containing text to tokenize
    /// @return JSON string containing token data
    /// @details Forwards tokenization to appropriate backend
    std::string tokenize_json(const json &data) override;

    /// @brief Convert tokens to text (override)
    /// @param data JSON object containing tokens to detokenize
    /// @return Detokenized text string
    /// @details Forwards detokenization to appropriate backend
    std::string detokenize_json(const json &data) override;

    /// @brief Generate embeddings (override)
    /// @param data JSON object containing text to embed
    /// @return JSON string with embedding vectors
    /// @details Forwards embedding generation to appropriate backend
    std::string embeddings_json(const json &data) override;

    /// @brief Generate text completion (override)
    /// @param data JSON object with prompt and parameters
    /// @param callback Optional callback for streaming responses
    /// @param callbackWithJSON Whether callback receives JSON format
    /// @return Generated completion text or JSON
    /// @details Forwards completion generation to appropriate backend with streaming support
    std::string completion_json(const json &data, CharArrayFn callback = nullptr, bool callbackWithJSON = true) override;

    /// @brief Manage processing slots (override)
    /// @param data JSON object with slot operation parameters
    /// @return JSON response string
    /// @details Forwards slot operations to appropriate backend
    std::string slot_json(const json &data) override;

    /// @brief Cancel running request (override)
    /// @param data JSON object with cancellation parameters
    /// @details Forwards cancellation request to appropriate backend
    void cancel_json(const json &data) override;

    /// @brief Get available processing slot (override)
    /// @return Available slot ID or -1 if none available
    /// @details Forwards slot availability check to appropriate backend
    int get_next_available_slot() override;
    //=================================== LLM METHODS END ===================================//

private:
    // Local LLM members
    LLMProvider *llm = nullptr; ///< Pointer to local LLM provider (null for remote clients)
    httplib::Client *client = nullptr;
    httplib::SSLClient *sslClient = nullptr;
    bool use_ssl = false;

    // Remote LLM members
    std::string url = "";      ///< Server URL for remote clients
    int port = -1;             ///< Server port for remote clients
    std::string API_key = "";  ///< API key for accessing remote server
    std::string SSL_cert = ""; ///< SSL certificate path for remote clients

    /// @brief Send HTTP POST request to remote server
    /// @param path API endpoint path
    /// @param payload JSON request payload
    /// @param callback Optional streaming callback function
    /// @param callbackWithJSON Whether callback receives JSON format
    /// @return HTTP response body
    /// @details Internal method for communicating with remote LLM servers
    std::string post_request(const std::string &path, const json &payload, CharArrayFn callback = nullptr, bool callbackWithJSON = true);
};

/// @ingroup c_api
/// @{

extern "C"
{
    /// @brief Set SSL certificate (C API)
    /// @param llm LLMClient instance pointer
    /// @param SSL_cert Path to SSL certificate file
    /// @details Configure SSL certificate for remote client connections
    UNDREAMAI_API void LLMClient_Set_SSL(LLMClient *llm, const char *SSL_cert);

    /// @brief Construct local LLMClient (C API)
    /// @param llm LLMProvider instance to wrap
    /// @return Pointer to new LLMClient instance
    /// @details Creates a client for local LLM provider access
    UNDREAMAI_API LLMClient *LLMClient_Construct(LLMProvider *llm);

    /// @brief Construct remote LLMClient (C API)
    /// @param url Server URL or hostname
    /// @param port Server port number
    /// @return Pointer to new LLMClient instance
    /// @details Creates a client for remote LLM server access
    UNDREAMAI_API LLMClient *LLMClient_Construct_Remote(const char *url, const int port, const char *API_key = "");
}

/// @}