#include "LLM_agent.h"
    
LLMAgent::LLMAgent(LLMLocal* llm_) : llm(llm_) { }

//================ API ================//

LLMAgent* LLMAgent_Construct(LLMLocal* llm)
{
    return new LLMAgent(llm);
}

const char* LLMAgent_Completion(LLMAgent* llm, const char* prompt, CharArrayFn callback, const char* params_as_json) {
    return LLM_Completion(llm, prompt, callback, llm->id_slot, params_as_json);
}

const char* LLMAgent_Completion_JSON(LLMAgent* llm, const char* prompt, CharArrayFn callback, const char* params_as_json) {
    return LLM_Completion_JSON(llm, prompt, callback, llm->id_slot, params_as_json);
}

const char* LLMAgent_Slot(LLMAgent* llm, const char* action, const char* filepath) {
    return LLM_Slot(llm, llm->id_slot, action, filepath);
}

void LLMAgent_Cancel(LLMAgent* llm) {
    LLM_Cancel(llm, llm->id_slot);
}

