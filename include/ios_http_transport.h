#pragma once
#include <string>
#include <vector>
#include <memory>

typedef void (*CharArrayFnWithContext)(const char *, void* context);

// Result structure
struct HttpResult {
    std::string body;
    int status_code;
    std::string error_message;
    bool success;
    
    HttpResult() : status_code(0), success(false) {}
};

class IOSHttpTransport {
public:
    IOSHttpTransport(const std::string &host, bool use_ssl, int port = -1);
    ~IOSHttpTransport();
    
    // POST request with context support
    // callback: receives null-terminated strings with context
    // cancel_flag: pointer to bool that can be set to true to cancel
    HttpResult post_request(
        const std::string &path,
        const std::string &body,
        const std::vector<std::pair<std::string, std::string>> &headers,
        CharArrayFnWithContext callback = nullptr,
        void* callback_context = nullptr,
        bool *cancel_flag = nullptr);
    
    // Configuration
    void set_timeout(double timeout_seconds);
    
    // Utility
    std::string get_last_error() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};