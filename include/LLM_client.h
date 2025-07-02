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
    std::string tokenize_json(const json& data) override;
    std::string detokenize_json(const json& data) override;
    std::string embeddings_json(const json& data) override;
    std::string completion_json(const json& data, CharArrayFn callback = nullptr, bool callbackWithJSON=true) override;
    std::string slot_json(const json& data) override;
    void cancel(int id_slot) override;
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
    std::string tokenize_json(const json& data) override;
    std::string detokenize_json(const json& data) override;
    std::string embeddings_json(const json& data) override;
    std::string completion_json(const json& data, CharArrayFn callback = nullptr, bool callbackWithJSON=true) override;
    //=================================== LLM METHODS END ===================================//

private:
    const std::string url;
    const int port;
    std::string SSL_cert = "";

    std::string post_request(const std::string& path, const json& payload, CharArrayFn callback = nullptr, bool callbackWithJSON=true);
};


extern "C" {
    UNDREAMAI_API LLMClient* LLMClient_Construct(LLMProvider* llm);

    UNDREAMAI_API LLMRemoteClient* LLMRemoteClient_Construct(const char* url, const int port);
};