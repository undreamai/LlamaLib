#pragma once

#include "LLMFunctions.h"

#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <iostream>
#include <sstream>
#include <curl/curl.h>

// DEFINE_HANDLE_JSON_JSON_FUNC(tokenize)
//   will be translated to implementation of:
// std::string handle_tokenize_json(const json data) override;

#define DEFINE_HANDLE_JSON_JSON_FUNC(FUNC_NAME)                               \
    inline std::string handle_##FUNC_NAME##_json(const json data) override {  \
        return post_request(url, port, #FUNC_NAME, data);                     \
    }

class RemoteLLMClient : public LLMFunctions {
private:
    std::string url;
    int port;

public:
    RemoteLLMClient(const std::string& url, int port);

    std::string post_request(const std::string& url, int port, const std::string& path, const std::string& payload);

    DEFINE_HANDLE_JSON_JSON_FUNC(tokenize)
    DEFINE_HANDLE_JSON_JSON_FUNC(detokenize)
};
