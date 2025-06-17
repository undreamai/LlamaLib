#pragma once

#include "LLM.h"

#include <openssl/err.h>
#include <openssl/ssl.h>
#include "error_handling.h"

#include "common.h"

struct server_context;

class UNDREAMAI_API LLMService : public LLMProvider {
    public:
        LLMService(const json& params);
        LLMService(const std::string& params);
        LLMService(const char* params);
        LLMService(int argc, char ** argv);
        ~LLMService();

        void init(int argc, char** argv);
        void init(const std::string& params);
        void init(const char* params);

        static EVP_PKEY* load_key(const std::string& key_str);
        static X509* load_cert(const std::string& cert_str);

        //=================================== LLM METHODS START ===================================//
        int status_code() override;
        std::string status_message() override;

        void start_server() override;
        void stop_server() override;
        void join_server() override;
        void start() override;
        void stop() override;
        void join_service() override;
        void set_SSL(const char* SSL_cert, const char* SSL_key) override;
        bool started() override;

        int embedding_size() override;
        //=================================== LLM METHODS END ===================================//

    protected:
        //=================================== LLM METHODS START ===================================//
        std::string tokenize_impl(const json& data) override;
        std::string detokenize_impl(const json& data) override;
        std::string embeddings_impl(const json& data, httplib::Response* res=nullptr, std::function<bool()> is_connection_closed = always_false) override;
        std::string lora_weight_impl(const json& data, httplib::Response* res=nullptr) override;
        std::string lora_list_impl() override;
        std::string completion_impl(const json& data, CharArrayFn callback=nullptr, httplib::Response* res=nullptr, std::function<bool()> is_connection_closed = always_false, int oaicompat = 0) override;
        std::string slot_impl(const json& data, httplib::Response* res=nullptr) override;
        void cancel_impl(int id_slot) override;
        //=================================== LLM METHODS END ===================================//

    private:
        common_params params;
        bool llama_backend_has_init;
        server_context* ctx_server = nullptr;
        std::thread service_thread;
        std::thread server_thread;
        std::unique_ptr<httplib::Server> svr;
        std::string SSL_cert = "";
        std::string SSL_key = "";
        std::mutex start_stop_mutex;

        std::vector<char*> jsonToArguments(const json& params);
        std::vector<std::string> splitArguments(const std::string& inputString);
        std::string completion_streaming(
            std::unordered_set<int> id_tasks,
            CharArrayFn callback=nullptr,
            httplib::DataSink* sink=nullptr,
            std::function<bool()> is_connection_closed = always_false
        );
        bool middleware_validate_api_key(const httplib::Request & req, httplib::Response & res);
};

class LLMServiceRegistry {
public:
    static LLMServiceRegistry& instance() {
        static LLMServiceRegistry registry;
        return registry;
    }

    void register_instance(LLMService* service) {
        std::lock_guard<std::mutex> lock(mutex_);
        instances_.push_back(service);
    }

    void unregister_instance(LLMService* service) {
        std::lock_guard<std::mutex> lock(mutex_);
        instances_.erase(std::remove(instances_.begin(), instances_.end(), service), instances_.end());
    }

    std::vector<LLMService*> get_instances() {
        std::lock_guard<std::mutex> lock(mutex_);
        return instances_;
    }

private:
    std::mutex mutex_;
    std::vector<LLMService*> instances_;

    LLMServiceRegistry() = default;
    ~LLMServiceRegistry() = default;
    LLMServiceRegistry(const LLMServiceRegistry&) = delete;
    LLMServiceRegistry& operator=(const LLMServiceRegistry&) = delete;
};

extern "C" {
    UNDREAMAI_API LLMService* LLMService_Construct(const char* params_string);
};