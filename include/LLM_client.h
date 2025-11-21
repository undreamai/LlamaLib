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
#include "completion_processor.h"

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
    LLMClient(const std::string &url, const int port, const std::string &API_key = "", const int max_retries = 5);

    /// @brief Destructor
    ~LLMClient();


    bool is_server_alive();

    /// @brief Configure SSL certificate for remote connections
    /// @param SSL_cert Path to SSL certificate file
    /// @details Only applicable for remote clients. Sets up SSL verification.
    void set_SSL(const char *SSL_cert);

    /// @brief Check if this is a remote client
    /// @return true if configured for remote access, false for local access
    /// @details Helper method to determine the client's connection type
    bool is_remote() const { return !url.empty() && port > -1; }

    //=================================== LLM METHODS START ===================================//
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

    /// @brief Generate text completion (override)
    /// @param data JSON object with prompt and parameters
    /// @param callback Optional callback for streaming responses
    /// @param callbackWithJSON Whether callback receives JSON format
    /// @return Generated completion text or JSON
    std::string completion_json(const json &data, CharArrayFn callback = nullptr, bool callbackWithJSON = true) override;

    /// @brief Apply template to messages
    /// @param messages JSON array of chat messages
    /// @return Formatted chat string
    std::string apply_template(const json &messages) override;

    /// @brief Cancel running request (override)
    /// @param data JSON object with cancellation parameters
    void cancel(int id_slot) override;

    /// @brief Get available processing slot (override)
    /// @return Available slot ID or -1 if none available
    int get_next_available_slot() override;
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
    int max_retries = 5;
    std::vector<bool*> active_requests;


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
    UNDREAMAI_API bool LLMClient_Is_Server_Alive(LLMClient *llm);

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