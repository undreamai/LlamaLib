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
    LLMClient(LLMProvider* llm);

protected:
    //=================================== LLM METHODS START ===================================//
    std::string tokenize_impl(const json& data) override;
    std::string detokenize_impl(const json& data) override;
    std::string embeddings_impl(const json& data, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false) override;
    std::string completion_impl(const json& data, CharArrayFn callback = nullptr, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false, int oaicompat = 0) override;
    std::string slot_impl(const json& data, httplib::Response* res = nullptr) override;
    void cancel_impl(int id_slot) override;
    //=================================== LLM METHODS END ===================================//

private:
    LLMProvider* llm = nullptr;
};


class UNDREAMAI_API LLMRemoteClient : public LLMRemote {
public:
    LLMRemoteClient(const std::string& url, const int port);

    static X509_STORE* load_cert(const std::string& cert_str);
    //=================================== LLM METHODS START ===================================//
    void set_SSL(const char* SSL_cert) override;
    //=================================== LLM METHODS END ===================================//

protected:
    //=================================== LLM METHODS START ===================================//
    std::string tokenize_impl(const json& data) override;
    std::string detokenize_impl(const json& data) override;
    std::string embeddings_impl(const json& data, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false) override;
    std::string completion_impl(const json& data, CharArrayFn callback = nullptr, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false, int oaicompat = 0) override;
    //=================================== LLM METHODS END ===================================//

private:
    const std::string url;
    const int port;
    std::string SSL_cert = "";

    std::string post_request(const std::string& path, const json& payload, CharArrayFn callback = nullptr);
};


extern "C" {
    UNDREAMAI_API LLMClient* LLMClient_Construct(LLMProvider* llm);

    UNDREAMAI_API LLMRemoteClient* LLMRemoteClient_Construct(const char* url, const int port);
};