#include "LLM_agent.h"
    
LLMAgent::LLMAgent(LLMLocal* llm_) : llm(llm_) {
    id_slot = llm->get_available_slot();
}

void LLMAgent::set_slot(int id_slot_)
{
    if (id_slot != -1)
    {
        if (LLMClient* client = dynamic_cast<LLMClient*>(llm)) {
            if (client->is_remote())
            {
                id_slot_ = -1;
                std::cerr << "Remote clients can only use id_slot -1" << std::endl;
            }
        }
    }
    id_slot = id_slot_;
}

//================ API ================//

LLMAgent* LLMAgent_Construct(LLMLocal* llm)
{
    return new LLMAgent(llm);
}

const char* LLMAgent_Completion(LLMAgent* llm, const char* prompt, CharArrayFn callback, const char* params_as_json) {
    return LLM_Completion(llm, prompt, callback, llm->get_slot(), params_as_json);
}

const char* LLMAgent_Completion_JSON(LLMAgent* llm, const char* prompt, CharArrayFn callback, const char* params_as_json) {
    return LLM_Completion_JSON(llm, prompt, callback, llm->get_slot(), params_as_json);
}

const char* LLMAgent_Slot(LLMAgent* llm, const char* action, const char* filepath) {
    return LLM_Slot(llm, llm->get_slot(), action, filepath);
}

void LLMAgent_Cancel(LLMAgent* llm) {
    LLM_Cancel(llm, llm->get_slot());
}

