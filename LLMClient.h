#pragma once

#include <string>
#include <vector>

class LLMClient {
public:
    virtual ~LLMClient() = default;

    virtual std::vector<int> handle_tokenize(const std::string& query) = 0;
};

class RemoteLLMClient : public LLMClient {
public:
    RemoteLLMClient(const std::string& url, int port);
    std::vector<int> handle_tokenize(const std::string& query) override;

private:
    std::string build_tokenize_json(const std::string& content);
    std::string url;
    int port;
};
