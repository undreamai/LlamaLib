#pragma once

#include "LLM.h"

#include "stringwrapper.h"
#include "logging.h"
//#include "server.cpp"
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    #include <openssl/err.h>
    #include <openssl/ssl.h>
#endif
#include "error_handling.h"

#include "common.h"

struct server_context;

class UNDREAMAI_API LLMService : public LLMProvider {
    public:
        LLMService(std::string params_string);
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
        std::string handle_embeddings_impl(const json& data, httplib::Response* res=nullptr, std::function<bool()> is_connection_closed = always_true) override;
        std::string handle_lora_adapters_apply_impl(const json& data, httplib::Response* res=nullptr) override;
        std::string handle_lora_adapters_list_impl() override;
        std::string handle_completions_impl(const json& data, StringWrapper* stringWrapper=nullptr, httplib::Response* res=nullptr, std::function<bool()> is_connection_closed = always_true, int oaicompat = 0) override;
        std::string handle_slots_action_impl(const json& data, httplib::Response* res=nullptr) override;
        void handle_cancel_action(int id_slot) override;
        //================ LLM ================//

        void start_server();
        void stop_server();
        void start_service();
        void stop_service();
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
            StringWrapper* stringWrapper=nullptr,
            httplib::DataSink* sink=nullptr,
            std::function<bool()> is_connection_closed = always_true
        );
        bool middleware_validate_api_key(const httplib::Request & req, httplib::Response & res);
        void register_signal_handling();
        void unregister_signal_handling();
        void release_slot(server_slot& slot);
};

static std::vector<LLMService*> llm_instances;
static std::mutex llm_mutex;

extern "C" {
    UNDREAMAI_API LLMService* LLM_Construct(const char* params_string);
    UNDREAMAI_API const void LLM_Delete(LLMService* llm);
    UNDREAMAI_API const void LLM_Start(LLMService* llm);
    UNDREAMAI_API const bool LLM_Started(LLMService* llm);
    UNDREAMAI_API const void LLM_Stop(LLMService* llm);
    UNDREAMAI_API const void LLM_StartServer(LLMService* llm);
    UNDREAMAI_API const void LLM_StopServer(LLMService* llm);
    UNDREAMAI_API const void LLM_SetSSL(LLMService* llm, const char* SSL_cert, const char* SSL_key);
    UNDREAMAI_API const int LLM_Status(LLMService* llm, StringWrapper* wrapper);
    UNDREAMAI_API const int LLM_Embedding_Size(LLMService* llm);
	
#ifdef _DEBUG
    UNDREAMAI_API const bool LLM_IsDebuggerAttached(void);
#endif
};