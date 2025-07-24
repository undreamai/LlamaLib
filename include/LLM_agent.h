#pragma once

#include "LLM.h"
#include "LLM_client.h"

struct UNDREAMAI_API ChatMessage {
    std::string role;
    std::string content;

    ChatMessage() = default;
    ChatMessage(const std::string& role_, const std::string& content_)
        : role(role_), content(content_) {}

    // Convert to JSON
    json to_json() const {
        return json{{"role", role}, {"content", content}};
    }

    // Create from JSON
    static ChatMessage from_json(const json& j) {
        return ChatMessage(j.at("role").get<std::string>(), j.at("content").get<std::string>());
    }

    bool operator==(const ChatMessage& other) const {
        return role == other.role && content == other.content;
    }
};

class UNDREAMAI_API LLMAgent : public LLMLocal {
public:
    LLMAgent(LLMLocal* llm, const std::string& system_prompt = "", const std::string& user_role = "user", const std::string& assistant_role = "assistant");

    inline int get_slot() { return id_slot; }
    void set_slot(int id_slot);

    // set variables
    void set_user_role(const std::string& user_role_) { user_role = user_role_; }
    std::string get_user_role() const { return user_role; }
    void set_assistant_role(const std::string& assistant_role_) { assistant_role = assistant_role_; }
    std::string get_assistant_role() const { return assistant_role; }
    void set_system_prompt(const std::string& system_prompt_){ system_prompt = system_prompt_; clear_history(); }
    std::string get_system_prompt() const { return system_prompt; }
    void set_history(const json& history_){ history = history_; }
    json get_history() const { return history; }

    // History management
    void clear_history();
    void add_message(const std::string& role, const std::string& content);
    void remove_last_message();
    void save_history(const std::string& filepath) const;
    void load_history(const std::string& filepath);
    size_t get_history_size() const { return history.size(); }

    // Chat functionality
    std::string chat(const std::string& user_prompt, bool add_to_history = true, CharArrayFn callback = nullptr, const json& params_json = json({}), bool return_response_json=false);

    //=================================== Reimplement methods with id_slot ===================================//
    virtual json build_completion_json(const std::string& prompt, const json& params) { return llm->build_completion_json(prompt, this->id_slot, params); }
    virtual std::string completion(const std::string& prompt, CharArrayFn callback=nullptr, const json& params_json=json({}), bool return_response_json=false) {
        return llm->completion(prompt, callback, this->id_slot, params_json, return_response_json);
    }
    
    virtual json build_slot_json(const std::string& action, const std::string& filepath) { return llm->build_slot_json(this->id_slot, action, filepath); }
    virtual std::string slot(const std::string& action, const std::string& filepath) { return llm->slot(this->id_slot, action, filepath); }

    virtual json build_cancel_json() { return llm->build_cancel_json(this->id_slot); }
    virtual void cancel() { llm->cancel(this->id_slot); }
    //=================================== Reimplement methods with id_slot ===================================//

protected:
    //=================================== LLM METHODS START ===================================//
    std::string get_template_json() override { return llm->get_template_json(); }
    std::string apply_template_json(const json& data) override { return llm->apply_template_json(data); }
    std::string tokenize_json(const json& data) override { return llm->tokenize_json(data); }
    std::string detokenize_json(const json& data) override { return llm->detokenize_json(data); }
    std::string embeddings_json(const json& data) override { return llm->embeddings_json(data); }
    std::string completion_json(const json& data, CharArrayFn callback = nullptr, bool callbackWithJSON=true) override { return llm->completion_json(data, callback, callbackWithJSON); }
    std::string slot_json(const json& data) override { return llm->slot_json(data); }
    void cancel_json(const json& data) override { return llm->cancel_json(data); }
    //=================================== LLM METHODS END ===================================//

private:
    LLMLocal* llm = nullptr;
    int id_slot = -1;
    std::string system_prompt = "";
    std::string user_role = "user";
    std::string assistant_role = "assistant";
    std::string system_role = "system";
    json history;

    //=================================== Hide methods with id_slot ===================================//
    json build_completion_json(const std::string& prompt, int id_slot, const json& params) override { return build_completion_json(prompt, params); }
    std::string completion(const std::string& prompt, CharArrayFn callback=nullptr, int id_slot=-1, const json& params_json=json({}), bool return_response_json=false) override {
        return completion(prompt, callback, params_json, return_response_json);
    }

    json build_slot_json(int id_slot, const std::string& action, const std::string& filepath) override { return build_slot_json(action, filepath); }
    std::string slot(int id_slot, const std::string& action, const std::string& filepath) override { return slot(action, filepath); }

    json build_cancel_json(int id_slot) override { return build_cancel_json(); }
    void cancel(int id_slot) override { cancel(); }

    int get_available_slot() override { return llm->get_available_slot(); }
    //=================================== Hide methods with id_slot ===================================//
};

extern "C" {
    UNDREAMAI_API LLMAgent* LLMAgent_Construct(LLMLocal* llm, const char* system_prompt = "", const char* user_role = "user", const char* assistant_role = "assistant");

    UNDREAMAI_API void LLMAgent_Set_User_Role(LLMAgent* llm, const char* user_role);
    UNDREAMAI_API const char* LLMAgent_Get_User_Role(LLMAgent* llm);
    UNDREAMAI_API void LLMAgent_Set_Assistant_Role(LLMAgent* llm, const char* assistant_role);
    UNDREAMAI_API const char* LLMAgent_Get_Assistant_Role(LLMAgent* llm);
    UNDREAMAI_API void LLMAgent_Set_System_Prompt(LLMAgent* llm, const char* system_prompt);
    UNDREAMAI_API const char* LLMAgent_Get_System_Prompt(LLMAgent* llm);

    UNDREAMAI_API const char* LLMAgent_Chat(LLMAgent* llm, const char* user_prompt, bool add_to_history = true, CharArrayFn callback = nullptr, const char* params_json = "{}", bool return_response_json=false);
    UNDREAMAI_API const char* LLMAgent_Completion(LLMAgent* llm, const char* prompt, CharArrayFn callback=nullptr, const char* params_json="{}", bool return_response_json=false);
    UNDREAMAI_API const char* LLMAgent_Slot(LLMAgent* llm, const char* action, const char* filepath);
    UNDREAMAI_API void LLMAgent_Cancel(LLMAgent* llm);

    UNDREAMAI_API void LLMAgent_Clear_History(LLMAgent* llm);
    UNDREAMAI_API const char* LLMAgent_Get_History(LLMAgent* llm);
    UNDREAMAI_API void LLMAgent_Set_History(LLMAgent* llm, const char* history_json);
    UNDREAMAI_API void LLMAgent_Add_Message(LLMAgent* llm, const char* role, const char* content);
    UNDREAMAI_API void LLMAgent_Remove_Last_Message(LLMAgent* llm);
    UNDREAMAI_API void LLMAgent_Save_History(LLMAgent* llm, const char* filepath);
    UNDREAMAI_API void LLMAgent_Load_History(LLMAgent* llm, const char* filepath);
    UNDREAMAI_API size_t LLMAgent_Get_History_Size(LLMAgent* llm);
};