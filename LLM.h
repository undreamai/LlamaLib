#pragma once

#include <string>
#include "logging.h"
#include "json.hpp"
#include "httplib.h"

using json = nlohmann::ordered_json;

static bool always_false()
{
    return false;
}

static std::string args_to_command(int argc, char** argv)
{
    std::string command = "";
    for (int i = 1; i < argc; ++i) {
        command += argv[i];
        if (i < argc - 1) command += " ";
    }
    return command;
}

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
public:
    virtual std::string handle_tokenize_impl(const json& data) = 0;
    virtual std::string handle_detokenize_impl(const json& data) = 0;
    virtual std::string handle_embeddings_impl(const json& data, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false) = 0;
    virtual std::string handle_completions_impl(const json& data, CharArrayFn callback = nullptr, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false, int oaicompat = 0) = 0;

    virtual std::string handle_tokenize_json(const json& data);
    virtual json build_tokenize_json(const std::string& query);
    virtual std::vector<int> parse_tokenize_json(const json& result);
    virtual std::vector<int> handle_tokenize(const json& data);
    virtual std::string handle_tokenize_json(const std::string& query);
    virtual std::vector<int> handle_tokenize(const std::string& query);
    virtual std::string handle_tokenize_json(const char* query);
    virtual std::vector<int> handle_tokenize(const char* query);

    virtual std::string handle_detokenize_json(const json& data);
    virtual json build_detokenize_json(const std::vector<int32_t>& tokens);
    virtual std::string parse_detokenize_json(const json& result);
    virtual std::string handle_detokenize_json(const std::vector<int32_t>& tokens);
    virtual std::string handle_detokenize(const json& data);
    virtual std::string handle_detokenize(const std::vector<int32_t>& tokens);

    virtual std::string handle_embeddings_json(const json& data, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false);
    virtual json build_embeddings_json(const std::string& query);
    virtual std::vector<float> parse_embeddings_json(const json& result);
    virtual std::string handle_embeddings_json(const std::string& query, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false);
    virtual std::string handle_embeddings_json(const char* query, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false);
    virtual std::vector<float> handle_embeddings(const json& data, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false);
    virtual std::vector<float> handle_embeddings(const std::string& query, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false);
    virtual std::vector<float> handle_embeddings(const char* query, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false);

    virtual std::string handle_completions_json(const json& data, CharArrayFn callback = nullptr, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false, int oaicompat = 0);
    virtual json build_completions_json(const std::string& prompt, int id_slot, const json& params);
    virtual std::string parse_completions_json(const json& result);
    virtual std::string handle_completions_json(const std::string& prompt, int id_slot, const json& params, CharArrayFn callback = nullptr, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false, int oaicompat = 0);
    virtual std::string handle_completions(const json& data, CharArrayFn callback = nullptr, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false, int oaicompat = 0);
    virtual std::string handle_completions(const std::string& prompt, int id_slot, const json& params, CharArrayFn callback = nullptr, httplib::Response* res = nullptr, std::function<bool()> is_connection_closed = always_false, int oaicompat = 0);
};

class UNDREAMAI_API LLMWithSlot : public LLM {
public:
    virtual std::string handle_slots_action_impl(const json& data, httplib::Response* res = nullptr) = 0;
    virtual void handle_cancel_action_impl(int id_slot) = 0;

    virtual std::string handle_slots_action_json(const json& data, httplib::Response* res = nullptr);
    virtual json build_slots_action_json(int id_slot, std::string action, std::string filepath);
    virtual std::string parse_slots_action_json(const json& result);
    virtual std::string handle_slots_action_json(int id_slot, std::string action, std::string filepath, httplib::Response* res = nullptr);
    virtual std::string handle_slots_action(const json& data, httplib::Response* res = nullptr);
    virtual std::string handle_slots_action(int id_slot, std::string action, std::string filepath, httplib::Response* res = nullptr);
};

class UNDREAMAI_API LLMFull : public LLMWithSlot {
public:
    virtual std::string handle_lora_adapters_apply_impl(const json& data, httplib::Response* res = nullptr) = 0;
    virtual std::string handle_lora_adapters_list_impl() = 0;

    virtual std::string handle_lora_adapters_apply_json(const json& data, httplib::Response* res = nullptr);
    virtual json build_lora_adapters_apply_json(const std::vector<LoraIdScale>& loras);
    virtual bool parse_lora_adapters_apply_json(const json& result);
    virtual std::string handle_lora_adapters_apply_json(const std::vector<LoraIdScale>& loras, httplib::Response* res = nullptr);
    virtual bool handle_lora_adapters_apply(const json& data, httplib::Response* res = nullptr);
    virtual bool handle_lora_adapters_apply(const std::vector<LoraIdScale>& loras, httplib::Response* res = nullptr);

    virtual std::string handle_lora_adapters_list_json();
    virtual std::vector<LoraIdScalePath> parse_lora_adapters_list_json(const json& result);
    virtual std::vector<LoraIdScalePath> handle_lora_adapters_list();
};

class UNDREAMAI_API LLMProvider : public LLMFull {
public:
    virtual int get_status() = 0;
    virtual std::string get_status_message() = 0;

    virtual void start_server() = 0;
    virtual void stop_server() = 0;
    virtual void join_server() = 0;
    virtual void start_service() = 0;
    virtual void stop_service() = 0;
    virtual void join_service() = 0;
    virtual void set_SSL(const char* SSL_cert, const char* SSL_key) = 0;
    virtual bool is_running() = 0;

    virtual int embedding_size() = 0;
};

extern "C" {
    UNDREAMAI_API const int LLM_Test();
    UNDREAMAI_API const char* LLM_Tokenize(LLM* llm, const char* json_data);
    UNDREAMAI_API const char* LLM_Detokenize(LLM* llm, const char* json_data);
    UNDREAMAI_API const char* LLM_Embeddings(LLM* llm, const char* json_data);
    UNDREAMAI_API const char* LLM_Completion(LLM* llm, const char* json_data, CharArrayFn callback);

    UNDREAMAI_API const char* LLM_Slot(LLMWithSlot* llm, const char* json_data);
    UNDREAMAI_API void LLM_Cancel(LLMWithSlot* llm, int id_slot);

    UNDREAMAI_API const char* LLM_Lora_Weight(LLMFull* llm, const char* json_data);
    UNDREAMAI_API const char* LLM_Lora_List(LLMFull* llm);

    UNDREAMAI_API void LLM_Delete(LLMProvider* llm);
    UNDREAMAI_API void LLM_Start(LLMProvider* llm);
    UNDREAMAI_API const bool LLM_Started(LLMProvider* llm);
    UNDREAMAI_API void LLM_Stop(LLMProvider* llm);
    UNDREAMAI_API void LLM_Start_Server(LLMProvider* llm);
    UNDREAMAI_API void LLM_Stop_Server(LLMProvider* llm);
    UNDREAMAI_API void LLM_Join_Service(LLMProvider* llm);
    UNDREAMAI_API void LLM_Join_Server(LLMProvider* llm);
    UNDREAMAI_API void LLM_SetSSL(LLMProvider* llm, const char* SSL_cert, const char* SSL_key);
    UNDREAMAI_API const int LLM_Status_Code(LLMProvider* llm);
    UNDREAMAI_API const char* LLM_Status_Message(LLMProvider* llm);
    UNDREAMAI_API const int LLM_Embedding_Size(LLMProvider* llm);
};