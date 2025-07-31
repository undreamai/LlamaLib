#pragma once

#include "LLM.h"

#define LLAMALIB_INF(...) LOG_TMPL(GGML_LOG_LEVEL_INFO, -1, __VA_ARGS__)

struct common_params;
struct server_context;

class UNDREAMAI_API LLMService : public LLMProvider
{
public:
    LLMService();
    LLMService(const std::string &model_path, int num_threads = -1, int num_GPU_layers = 0, int num_parallel = 1, bool flash_attention = false, int context_size = 4096, int batch_size = 2048, bool embedding_only = false, const std::vector<std::string> &lora_paths = {});
    ~LLMService();

    static LLMService *from_params(const json &params_json);
    static LLMService *from_command(const std::string &command);
    static LLMService *from_command(int argc, char **argv);

    static std::vector<char *> jsonToArguments(const json &params_json);

    void init(int argc, char **argv);
    void init(const std::string &params_string);
    void init(const char *params_string);

    //=================================== LLM METHODS START ===================================//
    void debug(int debug_level) override;
    void logging_callback(CharArrayFn callback) override;

    std::string get_template_json() override;
    void set_template_json(const json &data) override;
    std::string apply_template_json(const json &data) override;
    std::string tokenize_json(const json &data) override;
    std::string detokenize_json(const json &data) override;
    std::string embeddings_json(const json &data) override;
    std::string lora_weight_json(const json &data) override;
    std::string lora_list_json() override;
    std::string completion_json(const json &data, CharArrayFn callback = nullptr, bool callbackWithJSON = true) override;
    std::string slot_json(const json &data) override;
    void cancel_json(const json &data) override;

    void start_server(const std::string &host = "0.0.0.0", int port = 0, const std::string &API_key = "") override;
    void stop_server() override;
    void join_server() override;
    void start() override;
    void stop() override;
    void join_service() override;
    void set_SSL(const std::string &SSL_cert, const std::string &SSL_key) override;
    bool started() override;
    int embedding_size() override;
    int get_available_slot() override;
    //=================================== LLM METHODS END ===================================//

protected:
    std::string embeddings_json(const json &data, httplib::Response *res, std::function<bool()> is_connection_closed = always_false);
    std::string lora_weight_json(const json &data, httplib::Response *res);
    std::string completion_json(const json &data, CharArrayFn callback, bool callbackWithJSON, httplib::Response *res, std::function<bool()> is_connection_closed = always_false, int oaicompat = 0);
    std::string slot_json(const json &data, httplib::Response *res);

private:
    common_params *params;
    bool llama_backend_has_init;
    server_context *ctx_server = nullptr;
    std::unique_ptr<httplib::Server> svr;
    std::string SSL_cert = "";
    std::string SSL_key = "";

    std::mutex start_stop_mutex;
    std::thread service_thread;
    std::condition_variable service_stopped_cv;
    bool service_stopped = false;
    std::thread server_thread;
    std::condition_variable server_stopped_cv;
    bool server_stopped = false;

    std::vector<std::string> splitArguments(const std::string &inputString);
    void init_template();
    const char *detect_chat_template();
    std::string completion_streaming(
        std::unordered_set<int> id_tasks,
        CharArrayFn callback = nullptr,
        bool callbackWithJSON = true,
        bool return_tokens = false,
        httplib::DataSink *sink = nullptr,
        std::function<bool()> is_connection_closed = always_false);
    bool middleware_validate_api_key(const httplib::Request &req, httplib::Response &res);
};

extern "C"
{
    UNDREAMAI_API void LLMService_Registry(LLMProviderRegistry *existing_instance);
    UNDREAMAI_API LLMService *LLMService_Construct(const char *model_path, int num_threads = -1, int num_GPU_layers = 0, int num_parallel = 1, bool flash_attention = false, int context_size = 4096, int batch_size = 2048, bool embedding_only = false, int lora_count = 0, const char **lora_paths = nullptr);
    UNDREAMAI_API LLMService *LLMService_From_Command(const char *params_string);
};