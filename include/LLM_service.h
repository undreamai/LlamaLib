#pragma once

#include "LLM.h"

#include <openssl/err.h>
#include <openssl/ssl.h>

#include "common.h"

struct server_context;

class UNDREAMAI_API LLMService : public LLMProvider {
    public:
        LLMService();
        LLMService(const std::string& model_path, int num_threads=-1, int num_GPU_layers=0, int num_parallel=1, bool flash_attention=false, int context_size=4096, int batch_size=2048, bool embedding_only=false, const std::vector<std::string>& lora_paths = {});
        ~LLMService();

        static LLMService* from_params(const json& params);
        static LLMService* from_command(const std::string& command);
        static LLMService* from_command(int argc, char ** argv);

        static EVP_PKEY* load_key(const std::string& key_str);
        static X509* load_cert(const std::string& cert_str);
        static std::vector<char*> jsonToArguments(const json& params);

        void init(int argc, char** argv);
        void init(const std::string& params);
        void init(const char* params);

        //=================================== LLM METHODS START ===================================//
        void start_server(const std::string& host="0.0.0.0", int port=0, const std::string& API_key="") override;
        void stop_server() override;
        void join_server() override;
        void start() override;
        void stop() override;
        void join_service() override;
        void set_SSL(const std::string& SSL_cert, const std::string& SSL_key) override;
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

        std::vector<std::string> splitArguments(const std::string& inputString);
        std::string completion_streaming(
            std::unordered_set<int> id_tasks,
            CharArrayFn callback=nullptr,
            httplib::DataSink* sink=nullptr,
            std::function<bool()> is_connection_closed = always_false
        );
        bool middleware_validate_api_key(const httplib::Request & req, httplib::Response & res);
};

extern "C" {
    UNDREAMAI_API LLMService* LLMService_Construct(const char* model_path, int num_threads=-1, int num_GPU_layers=0, int num_parallel=1, bool flash_attention=false, int context_size=4096, int batch_size=2048, bool embedding_only=false, int lora_count=0, const char** lora_paths=nullptr);
    UNDREAMAI_API LLMService* LLMService_From_Command(const char* params_string);
};