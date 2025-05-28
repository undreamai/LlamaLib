#pragma once

#include "stringwrapper.h"
#include "LLM.h"
#include "LLM_lib.h"

#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <iostream>
#include <sstream>
#include <curl/curl.h>

struct StreamingContext {
    std::string buffer;
    StringWrapper* stringWrapper = nullptr;
};

class UNDREAMAI_API LLMClient : public LLMWithSlot {
private:
    LLMProvider* llm = nullptr;

public:
    LLMClient(LLMProvider* llm);
    LLMClient(LLMLib* llmLib);

    //================ LLM ================//
    std::string handle_tokenize_impl(const json& data) override;
    std::string handle_detokenize_impl(const json& data) override;
    std::string handle_embeddings_impl(const json& data, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false) override;
    std::string handle_completions_impl(const json& data, StringWrapper* stringWrapper = nullptr, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false, int oaicompat = 0) override;
    std::string handle_slots_action_impl(const json& data, httplib::Response* res = nullptr) override;
    void handle_cancel_action_impl(int id_slot) override;
    //================ LLM ================//
};


class UNDREAMAI_API RemoteLLMClient : public LLM {
private:
    const std::string url;
    const int port;

    std::string post_request(const std::string& url, int port, const std::string& path, const json& payload, StringWrapper* stringWrapper = nullptr);

public:
    RemoteLLMClient(const std::string& url, const int port);

    //================ LLM ================//
    std::string handle_tokenize_impl(const json& data) override;
    std::string handle_detokenize_impl(const json& data) override;
    std::string handle_embeddings_impl(const json& data, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false) override;
    std::string handle_completions_impl(const json& data, StringWrapper* stringWrapper = nullptr, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false, int oaicompat = 0) override;
    //================ LLM ================//
};
