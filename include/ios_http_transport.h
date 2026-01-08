#pragma once
#include <string>
#include <functional>
#include <vector>
#include <memory>

using CharArrayFn = std::function<bool(const char*, size_t)>;

// Result structure for better error handling
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
    
    // Streaming POST request with improved error handling
    HttpResult post_request(
        const std::string &path,
        const std::string &body,
        const std::vector<std::pair<std::string, std::string>> &headers,
        CharArrayFn callback = nullptr,
        bool *cancel_flag = nullptr);
    
    // Configuration methods
    void set_timeout(double timeout_seconds);
    void set_certificate_pinning(const std::string &cert_pem);
    void enable_certificate_validation(bool enable);
    
    // Utility methods
    bool is_connected();
    std::string get_last_error() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};