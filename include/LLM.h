#pragma once

#include "logging.h"
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

class UNDREAMAI_API LLM {
protected:
    virtual std::string tokenize_impl(const json& data) = 0;
    virtual std::string detokenize_impl(const json& data) = 0;
    virtual std::string embeddings_impl(const json& data, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false) = 0;
    virtual std::string completion_impl(const json& data, CharArrayFn callback = nullptr, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false, int oaicompat = 0) = 0;

public:
    virtual json build_tokenize_json(const std::string& query);
    virtual std::vector<int> parse_tokenize_json(const json& result);
    virtual std::string tokenize_json(const json& data);
    virtual std::string tokenize_json(const std::string& query);
    virtual std::string tokenize_json(const char* query);
    virtual std::vector<int> tokenize(const json& data);
    virtual std::vector<int> tokenize(const std::string& query);
    virtual std::vector<int> tokenize(const char* query);

    virtual json build_detokenize_json(const std::vector<int32_t>& tokens);
    virtual std::string parse_detokenize_json(const json& result);
    virtual std::string detokenize_json(const json& data);
    virtual std::string detokenize_json(const std::vector<int32_t>& tokens);
    virtual std::string detokenize(const json& data);
    virtual std::string detokenize(const std::vector<int32_t>& tokens);

    virtual json build_embeddings_json(const std::string& query);
    virtual std::vector<float> parse_embeddings_json(const json& result);
    virtual std::string embeddings_json(const json& data, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false);
    virtual std::string embeddings_json(const std::string& query, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false);
    virtual std::string embeddings_json(const char* query, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false);
    virtual std::vector<float> embeddings(const json& data, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false);
    virtual std::vector<float> embeddings(const std::string& query, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false);
    virtual std::vector<float> embeddings(const char* query, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false);

    virtual json build_completion_json(const std::string& prompt, int id_slot, const json& params);
    virtual std::string parse_completion_json(const json& result);
    virtual std::string completion_json(const json& data, CharArrayFn callback = nullptr, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false, int oaicompat = 0);
    virtual std::string completion_json(const std::string& prompt, int id_slot, const json& params, CharArrayFn callback = nullptr, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false, int oaicompat = 0);
    virtual std::string completion(const json& data, CharArrayFn callback = nullptr, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false, int oaicompat = 0);
    virtual std::string completion(const std::string& prompt, int id_slot, const json& params, CharArrayFn callback = nullptr, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false, int oaicompat = 0);
};

class UNDREAMAI_API LLMLocal : public LLM {
protected:
    virtual std::string slot_impl(const json& data, httplib::Response* res = nullptr) = 0;
    virtual void cancel_impl(int id_slot) = 0;

public:
    virtual json build_slot_json(int id_slot, std::string action, std::string filepath);
    virtual std::string parse_slot_json(const json& result);
    virtual std::string slot_json(const json& data, httplib::Response* res = nullptr);
    virtual std::string slot_json(int id_slot, std::string action, std::string filepath, httplib::Response* res = nullptr);
    virtual std::string slot(const json& data, httplib::Response* res = nullptr);
    virtual std::string slot(int id_slot, std::string action, std::string filepath, httplib::Response* res = nullptr);

    virtual void cancel(int id_slot);
};

class UNDREAMAI_API LLMRemote: public LLM {
protected:
    virtual void set_SSL(const char* SSL_cert) = 0;
};

class UNDREAMAI_API LLMProvider : public LLMLocal {
protected:
    virtual std::string lora_weight_impl(const json& data, httplib::Response* res = nullptr) = 0;
    virtual std::string lora_list_impl() = 0;

public:
    virtual json build_lora_weight_json(const std::vector<LoraIdScale>& loras);
    virtual bool parse_lora_weight_json(const json& result);
    virtual std::string lora_weight_json(const json& data, httplib::Response* res = nullptr);
    virtual std::string lora_weight_json(const std::vector<LoraIdScale>& loras, httplib::Response* res = nullptr);
    virtual bool lora_weight(const json& data, httplib::Response* res = nullptr);
    virtual bool lora_weight(const std::vector<LoraIdScale>& loras, httplib::Response* res = nullptr);

    virtual std::string lora_list_json();
    virtual std::vector<LoraIdScalePath> parse_lora_list_json(const json& result);
    virtual std::vector<LoraIdScalePath> lora_list();

    virtual int status_code() = 0;
    virtual std::string status_message() = 0;

    virtual void start_server() = 0;
    virtual void stop_server() = 0;
    virtual void join_server() = 0;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void join_service() = 0;
    virtual void set_SSL(const char* SSL_cert, const char* SSL_key) = 0;
    virtual bool started() = 0;

    virtual int embedding_size() = 0;
};

extern "C" {
    UNDREAMAI_API const char* LLM_Tokenize(LLM* llm, const char* json_data);
    UNDREAMAI_API const char* LLM_Detokenize(LLM* llm, const char* json_data);
    UNDREAMAI_API const char* LLM_Embeddings(LLM* llm, const char* json_data);
    UNDREAMAI_API const char* LLM_Completion(LLM* llm, const char* json_data, CharArrayFn callback);

    UNDREAMAI_API const char* LLM_Slot(LLMLocal* llm, const char* json_data);
    UNDREAMAI_API void LLM_Cancel(LLMLocal* llm, int id_slot);

    UNDREAMAI_API const char* LLM_Lora_Weight(LLMProvider* llm, const char* json_data);
    UNDREAMAI_API const char* LLM_Lora_List(LLMProvider* llm);
    UNDREAMAI_API void LLM_Delete(LLMProvider* llm);
    UNDREAMAI_API void LLM_Start(LLMProvider* llm);
    UNDREAMAI_API const bool LLM_Started(LLMProvider* llm);
    UNDREAMAI_API void LLM_Stop(LLMProvider* llm);
    UNDREAMAI_API void LLM_Start_Server(LLMProvider* llm);
    UNDREAMAI_API void LLM_Stop_Server(LLMProvider* llm);
    UNDREAMAI_API void LLM_Join_Service(LLMProvider* llm);
    UNDREAMAI_API void LLM_Join_Server(LLMProvider* llm);
    UNDREAMAI_API void LLM_Set_SSL(LLMProvider* llm, const char* SSL_cert, const char* SSL_key);
    UNDREAMAI_API const int LLM_Status_Code(LLMProvider* llm);
    UNDREAMAI_API const char* LLM_Status_Message(LLMProvider* llm);
    UNDREAMAI_API const int LLM_Embedding_Size(LLMProvider* llm);
};