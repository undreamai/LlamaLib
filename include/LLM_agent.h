#pragma once

#include "LLM.h"
#include "LLM_client.h"

class UNDREAMAI_API LLMAgent : public LLMLocal {
public:
    LLMAgent(LLMLocal* llm);

    inline int get_slot() { return id_slot; }
    void set_slot(int id_slot);

    //=================================== Reimplement methods with id_slot ===================================//
    virtual json build_completion_json(const std::string& prompt, const json& params) { return llm->build_completion_json(prompt, this->id_slot, params); }
    virtual std::string completion(const std::string& prompt, CharArrayFn callback=nullptr, const json& params_as_json=json({})) { return llm->completion(prompt, callback, this->id_slot, params_as_json); }
    
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

    //=================================== Hide methods with id_slot ===================================//
    json build_completion_json(const std::string& prompt, int id_slot, const json& params) override { return build_completion_json(prompt, params); }
    std::string completion(const std::string& prompt, CharArrayFn callback=nullptr, int id_slot=-1, const json& params_as_json=json({})) override { return completion(prompt, callback, params_as_json); }

    json build_slot_json(int id_slot, const std::string& action, const std::string& filepath) override { return build_slot_json(action, filepath); }
    std::string slot(int id_slot, const std::string& action, const std::string& filepath) override { return slot(action, filepath); }

    json build_cancel_json(int id_slot) override { return build_cancel_json(); }
    void cancel(int id_slot) override { cancel(); }

    int get_available_slot() override { return llm->get_available_slot(); }
    //=================================== Hide methods with id_slot ===================================//
};

extern "C" {
    UNDREAMAI_API LLMAgent* LLMAgent_Construct(LLMLocal* llm);
    UNDREAMAI_API const char* LLMAgent_Completion(LLMAgent* llm, const char* prompt, CharArrayFn callback=nullptr, const char* params_json="{}");
    UNDREAMAI_API const char* LLMAgent_Completion_JSON(LLMAgent* llm, const char* prompt, CharArrayFn callback=nullptr, const char* params_json="{}");
    UNDREAMAI_API const char* LLMAgent_Slot(LLMAgent* llm, const char* action, const char* filepath);
    UNDREAMAI_API void LLMAgent_Cancel(LLMAgent* llm);
};
