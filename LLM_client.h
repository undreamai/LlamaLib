#pragma once

#include "stringwrapper.h"
#include "LLM.h"
#include "dynamic_loader.h"

#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <iostream>
#include <sstream>
#include <curl/curl.h>

enum LLMClientMode {
    /*LLMOBJECT,
    LLMLIB,*/
    LOCAL,
    REMOTE
};

class LLMClient : public LLM {
private:
    LLMClientMode mode;
    //LLMService* llm = nullptr;
    LLMLib* llmLib = nullptr;
    std::string url;
    int port;
    StringWrapper* stringWrapper = nullptr;

public:
    LLMClient(LLMService* llm);
    LLMClient(LLMLib* llmLib);
    LLMClient(const std::string& url, int port);

    std::string post_request(const std::string& url, int port, const std::string& path, const std::string& payload);
    // Method to set specific function pointers if needed
    void setFunctionPointer(const std::string& funcName, void* funcPtr);


    //================ LLM ================//
    std::string handle_tokenize_json(const json& data) override;
    std::string handle_detokenize_json(const json& data) override;
    std::string handle_embeddings_json(const json& data, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_true) override;
    std::string handle_lora_adapters_apply_json(const json& data, httplib::Response* res = nullptr) override;
    std::string handle_lora_adapters_list_json() override;
    std::string handle_completions_json(const json& data, StringWrapper* stringWrapper = nullptr, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_true, int oaicompat = 0);
    std::string handle_slots_action_json(const json& data, httplib::Response* res = nullptr) override;
    void handle_cancel_action(int id_slot) override;
    //================ LLM ================//
};