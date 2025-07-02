#pragma once

#include "logging.h"
#include "error_handling.h"
// increase max payload length to allow use of larger context size
#define CPPHTTPLIB_FORM_URL_ENCODED_PAYLOAD_MAX_LENGTH 1048576
// disable Nagle's algorithm
#define CPPHTTPLIB_TCP_NODELAY true
#include "httplib.h"

struct LoraIdScale {
    int id;
    float scale;

    bool operator==(const LoraIdScale& other) const {
        return id == other.id && scale == other.scale;
    }
};

struct LoraIdScalePath {
    int id;
    float scale;
    std::string path;

    bool operator==(const LoraIdScalePath& other) const {
        return id == other.id && scale == other.scale && path == other.path;
    }
};

void ensure_error_handlers_initialized();

class UNDREAMAI_API LLM {
public:
    uint32_t seed = 0;
    int32_t n_predict = -1;
    int32_t n_keep = 0;
    float temperature = 0.80f;
    std::string json_schema = "";
    std::string grammar = "";

    virtual std::string tokenize_json(const json& data) = 0;
    virtual std::string detokenize_json(const json& data) = 0;
    virtual std::string embeddings_json(const json& data) = 0;
    virtual std::string completion_json(const json& data, CharArrayFn callback, bool callbackWithJSON) = 0;

    static bool has_gpu_layers(const std::string& command);
    static std::string LLM_args_to_command(const std::string& model_path, int num_threads=-1, int num_GPU_layers=0, int num_parallel=1, bool flash_attention=false, int context_size=4096, int batch_size=2048, bool embedding_only=false, const std::vector<std::string>& lora_paths = {});

    virtual json build_tokenize_json(const std::string& query);
    virtual std::vector<int> parse_tokenize_json(const json& result);
    virtual std::vector<int> tokenize(const std::string& query);

    virtual json build_detokenize_json(const std::vector<int32_t>& tokens);
    virtual std::string parse_detokenize_json(const json& result);
    virtual std::string detokenize(const std::vector<int32_t>& tokens);

    virtual json build_embeddings_json(const std::string& query);
    virtual std::vector<float> parse_embeddings_json(const json& result);
    virtual std::vector<float> embeddings(const std::string& query);

    virtual json build_completion_json(const std::string& prompt, int id_slot, const json& params=json({}));
    virtual std::string parse_completion_json(const json& result);
    virtual std::string completion(const std::string& prompt, int id_slot, CharArrayFn callback = nullptr, const json& params=json({}));
};

class UNDREAMAI_API LLMLocal : public LLM {
public:
    virtual std::string slot_json(const json& data) = 0;
    virtual void cancel(int id_slot) = 0;

    virtual json build_slot_json(int id_slot, const std::string& action, const std::string& filepath);
    virtual std::string parse_slot_json(const json& result);
    virtual std::string slot(int id_slot, const std::string& action, const std::string& filepath);
};

class UNDREAMAI_API LLMRemote: public LLM {
protected:
    virtual void set_SSL(const char* SSL_cert) = 0;
};

class UNDREAMAI_API LLMProvider : public LLMLocal {
public:
    virtual std::string lora_weight_json(const json& data) = 0;
    virtual std::string lora_list_json() = 0;

    virtual json build_lora_weight_json(const std::vector<LoraIdScale>& loras);
    virtual bool parse_lora_weight_json(const json& result);
    virtual bool lora_weight(const std::vector<LoraIdScale>& loras);

    virtual std::vector<LoraIdScalePath> parse_lora_list_json(const json& result);
    virtual std::vector<LoraIdScalePath> lora_list();

    virtual void start_server(const std::string& host="0.0.0.0", int port=0, const std::string& API_key="") = 0;
    virtual void stop_server() = 0;
    virtual void join_server() = 0;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void join_service() = 0;
    virtual void set_SSL(const std::string& SSL_cert, const std::string& SSL_key) = 0;
    virtual bool started() = 0;

    virtual int embedding_size() = 0;
};

class LLMProviderRegistry {
public:
    static LLMProviderRegistry& instance() {
        static LLMProviderRegistry registry;
        return registry;
    }

    void register_instance(LLMProvider* service) {
        std::lock_guard<std::mutex> lock(mutex_);
        instances_.push_back(service);
    }

    void unregister_instance(LLMProvider* service) {
        std::lock_guard<std::mutex> lock(mutex_);
        instances_.erase(std::remove(instances_.begin(), instances_.end(), service), instances_.end());
    }

    std::vector<LLMProvider*> get_instances() {
        std::lock_guard<std::mutex> lock(mutex_);
        return instances_;
    }

private:
    std::mutex mutex_;
    std::vector<LLMProvider*> instances_;

    LLMProviderRegistry() = default;
    ~LLMProviderRegistry() = default;
    LLMProviderRegistry(const LLMProviderRegistry&) = delete;
    LLMProviderRegistry& operator=(const LLMProviderRegistry&) = delete;
};


extern "C" {
    UNDREAMAI_API bool Has_GPU_Layers(const char* command);

    UNDREAMAI_API const char* LLM_Tokenize(LLM* llm, const char* json_data);
    UNDREAMAI_API const char* LLM_Detokenize(LLM* llm, const char* json_data);
    UNDREAMAI_API const char* LLM_Embeddings(LLM* llm, const char* json_data);
    UNDREAMAI_API const char* LLM_Completion(LLM* llm, const char* json_data, CharArrayFn callback=nullptr, bool callbackWithJSON=true);

    UNDREAMAI_API const char* LLM_Slot(LLMLocal* llm, const char* json_data);
    UNDREAMAI_API void LLM_Cancel(LLMLocal* llm, int id_slot);

    UNDREAMAI_API const char* LLM_Lora_Weight(LLMProvider* llm, const char* json_data);
    UNDREAMAI_API const char* LLM_Lora_List(LLMProvider* llm);
    UNDREAMAI_API void LLM_Delete(LLMProvider* llm);
    UNDREAMAI_API void LLM_Start(LLMProvider* llm);
    UNDREAMAI_API const bool LLM_Started(LLMProvider* llm);
    UNDREAMAI_API void LLM_Stop(LLMProvider* llm);
    UNDREAMAI_API void LLM_Start_Server(LLMProvider* llm, const char* host="0.0.0.0", int port=0, const char* API_key="");
    UNDREAMAI_API void LLM_Stop_Server(LLMProvider* llm);
    UNDREAMAI_API void LLM_Join_Service(LLMProvider* llm);
    UNDREAMAI_API void LLM_Join_Server(LLMProvider* llm);
    UNDREAMAI_API void LLM_Set_SSL(LLMProvider* llm, const char* SSL_cert, const char* SSL_key);
    UNDREAMAI_API const int LLM_Status_Code(LLMProvider* llm);
    UNDREAMAI_API const char* LLM_Status_Message(LLMProvider* llm);
    UNDREAMAI_API const int LLM_Embedding_Size(LLMProvider* llm);
};