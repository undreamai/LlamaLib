#pragma once

#include <vector>
#include <thread>
#include <chrono>
#include <iostream>
#include <sstream>
#include "LLM.h"

struct StreamingContext {
    std::string buffer;
    CharArrayFn callback = nullptr;
};


class UNDREAMAI_API LLMClient : public LLMLocal {
public:
    // Constructor for local LLM
    LLMClient(LLMProvider* llm);
    
    // Constructor for remote LLM
    LLMClient(const std::string& url, const int port);

    // SSL support (only for remote clients)
    void set_SSL(const char* SSL_cert);

    // Helper to determine if this is a remote client
    bool is_remote() const { return !url.empty() && port > -1; }

protected:
    //=================================== LLM METHODS START ===================================//
    std::string get_template_json() override;
    std::string apply_template_json(const json& data) override;
    std::string tokenize_json(const json& data) override;
    std::string detokenize_json(const json& data) override;
    std::string embeddings_json(const json& data) override;
    std::string completion_json(const json& data, CharArrayFn callback = nullptr, bool callbackWithJSON=true) override;
    std::string slot_json(const json& data) override;
    void cancel_json(const json& data) override;
    int get_available_slot() override;
    //=================================== LLM METHODS END ===================================//

private:
    // Local LLM members
    LLMProvider* llm = nullptr;
    
    // Remote LLM members
    std::string url = "";
    int port = -1;
    std::string SSL_cert = "";
    
    // Remote request helper
    std::string post_request(const std::string& path, const json& payload, CharArrayFn callback = nullptr, bool callbackWithJSON=true);
};

extern "C" {
    UNDREAMAI_API void LLMClient_Set_SSL(LLMClient* llm, const char* SSL_cert);
    UNDREAMAI_API LLMClient* LLMClient_Construct(LLMProvider* llm);
    UNDREAMAI_API LLMClient* LLMClient_Construct_Remote(const char* url, const int port);
};