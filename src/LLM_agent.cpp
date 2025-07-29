#include "LLM_agent.h"
#include <fstream>
#include <iostream>

LLMAgent::LLMAgent(LLMLocal* llm_, const std::string& system_prompt_, const std::string& user_role_, const std::string& assistant_role_)
    : llm(llm_), system_prompt(system_prompt_), user_role(user_role_), assistant_role(assistant_role_) {
    id_slot = llm->get_available_slot();
    clear_history();
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

void LLMAgent::clear_history()
{
    history = json::array();
    ChatMessage system_msg(system_role, system_prompt);
    history.push_back(system_msg.to_json());

    json working_history = history;
    ChatMessage user_msg(user_role, "");
    working_history.push_back(user_msg.to_json());
    n_keep = tokenize(apply_template(working_history)).size();
}

std::string LLMAgent::chat(const std::string& user_prompt, bool add_to_history, CharArrayFn callback, bool return_response_json)
{
    // Add user message to working history
    json working_history = history;
    ChatMessage user_msg(user_role, user_prompt);
    working_history.push_back(user_msg.to_json());

    // Apply template to get the formatted prompt
    std::string query_prompt = apply_template(working_history);

    // Call completion with the formatted prompt
    std::string response = completion(query_prompt, callback, return_response_json);
    std::string assistant_content = response;
    if (return_response_json) assistant_content = parse_completion_json(response);

    if (add_to_history) {
        history.push_back(user_msg.to_json());
        ChatMessage assistant_msg(assistant_role, assistant_content);
        history.push_back(assistant_msg.to_json());
    }

    return response;
}

void LLMAgent::add_message(const std::string& role, const std::string& content)
{
    ChatMessage msg(role, content);
    history.push_back(msg.to_json());
}

void LLMAgent::remove_last_message()
{
    if (!history.empty()) {
        history.erase(history.end() - 1);
    }
}

void LLMAgent::save_history(const std::string& filepath) const
{
    try {
        std::ofstream file(filepath);
        if (file.is_open()) {
            file << history.dump(4); // Pretty print with 4 spaces
            file.close();
        } else {
            std::cerr << "Unable to open file for writing: " << filepath << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error saving history to file: " << e.what() << std::endl;
    }
}

void LLMAgent::load_history(const std::string& filepath)
{
    try {
        std::ifstream file(filepath);
        if (file.is_open()) {
            json loaded_history;
            file >> loaded_history;
            file.close();

            if (loaded_history.is_array()) {
                history = loaded_history;
            } else {
                std::cerr << "Invalid history file format: expected JSON array" << std::endl;
            }
        } else {
            std::cerr << "Unable to open file for reading: " << filepath << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading history from file: " << e.what() << std::endl;
    }
}

//================ C API ================//

LLMAgent* LLMAgent_Construct(LLMLocal* llm, const char* system_prompt_, const char* user_role_, const char* assistant_role_)
{
    std::string system_prompt = system_prompt_ ? system_prompt_ : "";
    std::string user_role = user_role_ ? user_role_ : "user";
    std::string assistant_role = assistant_role_ ? assistant_role_ : "assistant";
    return new LLMAgent(llm, system_prompt, user_role, assistant_role);
}

const char* LLMAgent_Chat(LLMAgent* llm, const char* user_prompt, bool add_to_history, CharArrayFn callback, bool return_response_json)
{
    return stringToCharArray(llm->chat(user_prompt, add_to_history, callback, return_response_json));
}

// History management C API implementations
void LLMAgent_Clear_History(LLMAgent* llm)
{
    llm->clear_history();
}

void LLMAgent_Set_System_Prompt(LLMAgent* llm, const char* system_prompt)
{
    std::string sys_prompt = system_prompt ? system_prompt : "";
    llm->set_system_prompt(sys_prompt);
}

const char* LLMAgent_Get_System_Prompt(LLMAgent* llm)
{
    return stringToCharArray(llm->get_system_prompt());
}

void LLMAgent_Set_User_Role(LLMAgent* llm, const char* user_role_)
{
    std::string user_role = user_role_ ? user_role_ : "user";
    llm->set_user_role(user_role);
}

const char* LLMAgent_Get_User_Role(LLMAgent* llm)
{
    return stringToCharArray(llm->get_user_role());
}

void LLMAgent_Set_Assistant_Role(LLMAgent* llm, const char* assistant_role_)
{
    std::string assistant_role = assistant_role_ ? assistant_role_ : "assistant";
    llm->set_assistant_role(assistant_role);
}

const char* LLMAgent_Get_Assistant_Role(LLMAgent* llm)
{
    return stringToCharArray(llm->get_assistant_role());
}

const char* LLMAgent_Get_History(LLMAgent* llm)
{
    return stringToCharArray(llm->get_history().dump());
}

void LLMAgent_Set_Slot(LLMAgent* llm, int slot_id)
{
    llm->set_slot(slot_id);
}

int LLMAgent_Get_Slot(LLMAgent* llm)
{
    return llm->get_slot();
}

void LLMAgent_Set_History(LLMAgent* llm, const char* history_json)
{
    try {
        json history = json::parse(history_json ? history_json : "[]");
        if (!history.is_array()) std::cerr << "Expected JSON array for history." << std::endl;
        else llm->set_history(history);
    } catch (const std::exception& e) {
        std::cerr << "Error parsing history JSON: " << e.what() << std::endl;
    }
}

void LLMAgent_Add_Message(LLMAgent* llm, const char* role, const char* content)
{
    std::string role_str = role ? role : "";
    std::string content_str = content ? content : "";
    llm->add_message(role_str, content_str);
}

void LLMAgent_Remove_Last_Message(LLMAgent* llm)
{
    llm->remove_last_message();
}

void LLMAgent_Save_History(LLMAgent* llm, const char* filepath)
{
    std::string path = filepath ? filepath : "";
    if (!path.empty()) {
        llm->save_history(path);
    }
}

void LLMAgent_Load_History(LLMAgent* llm, const char* filepath)
{
    std::string path = filepath ? filepath : "";
    if (!path.empty()) {
        llm->load_history(path);
    }
}

size_t LLMAgent_Get_History_Size(LLMAgent* llm)
{
    return llm->get_history_size();
}
