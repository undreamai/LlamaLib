#pragma once

#include "LLM.h"

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    #include <openssl/err.h>
    #include <openssl/ssl.h>
#endif
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

#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        static EVP_PKEY *load_key(const std::string& key_str);
        static X509 *load_cert(const std::string& cert_str);
#endif
        static std::vector<std::string> splitArguments(const std::string& inputString);
        int get_status();
        std::string get_status_message();
        
        //================ LLM ================//
        std::string handle_tokenize_impl(const json& data) override;
        std::string handle_detokenize_impl(const json& data) override;
        std::string handle_embeddings_impl(const json& data, httplib::Response* res=nullptr, std::function<bool()> is_connection_closed = always_false) override;
        std::string handle_lora_adapters_apply_impl(const json& data, httplib::Response* res=nullptr) override;
        std::string handle_lora_adapters_list_impl() override;
        std::string handle_completions_impl(const json& data, CharArrayFn callback=nullptr, httplib::Response* res=nullptr, std::function<bool()> is_connection_closed = always_false, int oaicompat = 0) override;
        std::string handle_slots_action_impl(const json& data, httplib::Response* res=nullptr) override;
        void handle_cancel_action_impl(int id_slot) override;
        //================ LLM ================//

        void start_server();
        void stop_server();
        void join_server();
        void start_service();
        void stop_service();
        void join_service();
        void set_SSL(const char* SSL_cert, const char* SSL_key);
        bool is_running();

        int embedding_size();

    private:
        common_params params;
        bool llama_backend_has_init;
        server_context* ctx_server = nullptr;
        std::thread service_thread;
        std::thread server_thread;
        std::unique_ptr<httplib::Server> svr;
        std::string SSL_cert = "";
        std::string SSL_key = "";

        void init(int argc, char ** argv);
        std::string handle_completions_streaming(
            std::unordered_set<int> id_tasks,
            CharArrayFn callback=nullptr,
            httplib::DataSink* sink=nullptr,
            std::function<bool()> is_connection_closed = always_false
        );
        bool middleware_validate_api_key(const httplib::Request & req, httplib::Response & res);
        void register_signal_handling();
        void unregister_signal_handling();
        void release_slot(server_slot& slot);
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
    UNDREAMAI_API LLMService* LLM_Construct(const char* params_string);
    UNDREAMAI_API void LLM_Delete(LLMService* llm);
    UNDREAMAI_API void LLM_Start(LLMService* llm);
    UNDREAMAI_API const bool LLM_Started(LLMService* llm);
    UNDREAMAI_API void LLM_Stop(LLMService* llm);
    UNDREAMAI_API void LLM_StartServer(LLMService* llm);
    UNDREAMAI_API void LLM_StopServer(LLMService* llm);
    UNDREAMAI_API void LLM_Join_Service(LLMService* llm);
    UNDREAMAI_API void LLM_Join_Server(LLMService* llm);
    UNDREAMAI_API void LLM_SetSSL(LLMService* llm, const char* SSL_cert, const char* SSL_key);
    UNDREAMAI_API const int LLM_Status_Code(LLMService* llm);
    UNDREAMAI_API const char* LLM_Status_Message(LLMService* llm);
    UNDREAMAI_API const int LLM_Embedding_Size(LLMService* llm);
	
#ifdef _DEBUG
    UNDREAMAI_API const bool LLM_IsDebuggerAttached(void);
#endif
};