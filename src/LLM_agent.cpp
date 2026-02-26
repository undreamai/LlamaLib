#include "LLM_agent.h"
#include <fstream>
#include <iostream>

LLMAgent::LLMAgent(LLMLocal *llm_, const std::string &system_prompt_) : llm(llm_), system_prompt(system_prompt_)
{
    id_slot = llm->get_next_available_slot();
    clear_history();
}

void LLMAgent::set_slot(int id_slot_)
{
    if (id_slot != -1)
    {
        if (LLMClient *client = dynamic_cast<LLMClient *>(llm))
        {
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
    summary = "";
    n_keep = -1;
}

json LLMAgent::build_working_history(const std::string &user_prompt) const
{
    json working_history = json::array();
    std::string effective_system = system_prompt;
    if (!summary.empty())
        effective_system += "\n\n[Conversation summary]\n" + summary;
    working_history.push_back(ChatMessage(system_role, effective_system).to_json());
    for (const auto &m : history) working_history.push_back(m);
    if (!user_prompt.empty())
        working_history.push_back(ChatMessage(USER_ROLE, user_prompt).to_json());
    return working_history;
}

void LLMAgent::set_n_keep()
{
    try
    {
        n_keep = tokenize(apply_template(build_working_history(""))).size();
    } catch(...){ }
}

std::string LLMAgent::chat(const std::string &user_prompt, bool add_to_history, CharArrayFn callback, bool return_response_json, bool debug_prompt)
{
    if (n_keep == -1) set_n_keep();

    // Handle context overflow before sending
    if (overflow_strategy != ContextOverflowStrategy::None)
        handle_overflow(user_prompt);

    // Apply template to get the formatted prompt
    std::string query_prompt = apply_template(build_working_history(user_prompt));
    if (debug_prompt)
    {
        LLMProviderRegistry &registry = LLMProviderRegistry::instance();
        auto log_callback = registry.get_log_callback();
        if (log_callback != nullptr) log_callback(query_prompt.c_str());
    }

    // Call completion with the formatted prompt
    std::string response = completion(query_prompt, callback, return_response_json);
    std::string assistant_content = response;
    if (return_response_json)
        assistant_content = parse_completion_json(response);

    if (add_to_history)
    {
        history.push_back(ChatMessage(USER_ROLE, user_prompt).to_json());
        history.push_back(ChatMessage(ASSISTANT_ROLE, assistant_content).to_json());
    }

    return response;
}

void LLMAgent::add_message(const std::string &role, const std::string &content)
{
    ChatMessage msg(role, content);
    history.push_back(msg.to_json());
}

void LLMAgent::remove_last_message()
{
    if (!history.empty())
    {
        history.erase(history.end() - 1);
    }
}

void LLMAgent::save_history(const std::string &filepath) const
{
    try
    {
        std::ofstream file(filepath);
        if (file.is_open())
        {
            json save_data;
            save_data["history"] = history;
            save_data["summary"] = summary;
            file << save_data.dump(4);
            file.close();
        }
        else
        {
            std::cerr << "Unable to open file for writing: " << filepath << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error saving history to file: " << e.what() << std::endl;
    }
}

void LLMAgent::load_history(const std::string &filepath)
{
    try
    {
        std::ifstream file(filepath);
        if (file.is_open())
        {
            json data;
            file >> data;
            file.close();

            // New format: {"history": [...], "summary": "..."}
            if (data.is_object() && data.contains("history") && data["history"].is_array())
            {
                history = data["history"];
                summary = data.value("summary", "");
            }
            // Legacy format: plain JSON array (no summary)
            else if (data.is_array())
            {
                history = data;
                summary = "";
            }
            else
            {
                std::cerr << "Invalid history file format" << std::endl;
            }
        }
        else
        {
            std::cerr << "Unable to open file for reading: " << filepath << std::endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error loading history from file: " << e.what() << std::endl;
    }
}

bool LLMAgent::handle_overflow(const std::string &user_prompt)
{
    int ctx = get_slot_context_size();
    if (ctx <= 0) return false;

    int prompt_tokens = static_cast<int>(tokenize(apply_template(build_working_history(user_prompt))).size());
    if (prompt_tokens < ctx) return false;

    switch (overflow_strategy)
    {
        case ContextOverflowStrategy::Truncate:
            truncate_history(user_prompt);
            return true;
        case ContextOverflowStrategy::Summarize:
            summarize_history(user_prompt);
            return true;
        default:
            return false;
    }
}

void LLMAgent::truncate_history(const std::string &user_prompt)
{
    int ctx = get_slot_context_size();
    if (ctx <= 0 || history.empty()) return;

    int target_tokens = static_cast<int>(ctx * target_context_ratio);

    auto measure = [&]() -> int {
        return static_cast<int>(tokenize(apply_template(build_working_history(user_prompt))).size());
    };

    while (history.size() >= 2 && measure() > target_tokens)
        history.erase(history.begin(), history.begin() + 2);

    // Edge case: a single orphan message still overflows
    if (!history.empty() && measure() > target_tokens)
        history.erase(history.begin());
}

void LLMAgent::summarize_history(const std::string &user_prompt)
{
    if (history.empty()) return;
    int ctx = get_slot_context_size();
    if (ctx <= 0) return;

    // Build the prompt for a summary request, incorporating any prior rolling summary.
    auto build_summary_prompt = [&](const std::string &transcript) -> std::string {
        std::string query = summarize_prompt;
        if (!summary.empty())
            query += "Current summary:\n" + summary + "\n\n";
        query += "Messages:\n" + transcript;
        json msgs = json::array();
        msgs.push_back(ChatMessage(USER_ROLE, query).to_json());
        return apply_template(msgs);
    };

    try
    {
        // Walk history, flushing a summary call whenever the accumulating transcript
        // would itself overflow the context.
        std::string transcript;
        for (int i=0; i<history.size(); i+=2)
        {
            std::string line = "";
            for (int j=0; j<2; j++)
            {
                if (i+j >= history.size()-1) break;
                json msg = history[i+j];
                std::string role    = msg.at("role").get<std::string>();
                std::string content = msg.at("content").get<std::string>();
                line += role + ": " + content + "\n";
            }

            // Flush before appending if this line would push the prompt over the limit
            if (!transcript.empty() && static_cast<int>(tokenize(build_summary_prompt(transcript + line)).size()) >= ctx*0.75)
            {
                summary = completion(build_summary_prompt(transcript));
                transcript = "";
            }
            transcript += line;
        }
        if (!transcript.empty())
            summary = completion(build_summary_prompt(transcript));

        // History is now condensed into summary — clear the raw messages
        history = json::array();

        // The summary lives in the system message (injected by build_working_history).
        // If even system + summary + empty user message overflows the budget, discard it.
        int probe_tokens = static_cast<int>(tokenize(apply_template(build_working_history(user_prompt))).size());
        int target_tokens = static_cast<int>(ctx * target_context_ratio);
        if (probe_tokens > target_tokens)
        {
            std::cerr << "LLMAgent: summary itself exceeds context budget — discarding summary" << std::endl;
            summary = "";
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "LLMAgent: summarization failed (" << e.what() << "), falling back to truncation" << std::endl;
        history = json::array(); // clear history to avoid double-processing
        truncate_history(user_prompt);
    }
}

//================ C API ================//

LLMAgent *LLMAgent_Construct(LLMLocal *llm, const char *system_prompt_)
{
    std::string system_prompt = system_prompt_ ? system_prompt_ : "";
    return new LLMAgent(llm, system_prompt);
}

const char *LLMAgent_Chat(LLMAgent *llm, const char *user_prompt, bool add_to_history, CharArrayFn callback, bool return_response_json, bool debug_prompt)
{
    return stringToCharArray(llm->chat(user_prompt, add_to_history, callback, return_response_json, debug_prompt));
}

// History management C API implementations
void LLMAgent_Clear_History(LLMAgent *llm)
{
    llm->clear_history();
}

void LLMAgent_Set_System_Prompt(LLMAgent *llm, const char *system_prompt)
{
    std::string sys_prompt = system_prompt ? system_prompt : "";
    llm->set_system_prompt(sys_prompt);
}

const char *LLMAgent_Get_System_Prompt(LLMAgent *llm)
{
    return stringToCharArray(llm->get_system_prompt());
}

const char *LLMAgent_Get_History(LLMAgent *llm)
{
    return stringToCharArray(llm->get_history().dump());
}

void LLMAgent_Set_Slot(LLMAgent *llm, int slot_id)
{
    llm->set_slot(slot_id);
}

int LLMAgent_Get_Slot(LLMAgent *llm)
{
    return llm->get_slot();
}

void LLMAgent_Set_History(LLMAgent *llm, const char *history_json)
{
    try
    {
        json history = json::parse(history_json ? history_json : "[]");
        if (!history.is_array())
            std::cerr << "Expected JSON array for history." << std::endl;
        else
            llm->set_history(history);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error parsing history JSON: " << e.what() << std::endl;
    }
}

void LLMAgent_Add_User_Message(LLMAgent *llm, const char *content)
{
    llm->add_user_message(content ? content : "");
}

void LLMAgent_Add_Assistant_Message(LLMAgent *llm, const char *content)
{
    llm->add_assistant_message(content ? content : "");
}

void LLMAgent_Remove_Last_Message(LLMAgent *llm)
{
    llm->remove_last_message();
}

void LLMAgent_Save_History(LLMAgent *llm, const char *filepath)
{
    std::string path = filepath ? filepath : "";
    if (!path.empty())
    {
        llm->save_history(path);
    }
}

void LLMAgent_Load_History(LLMAgent *llm, const char *filepath)
{
    std::string path = filepath ? filepath : "";
    if (!path.empty())
    {
        llm->load_history(path);
    }
}

size_t LLMAgent_Get_History_Size(LLMAgent *llm)
{
    return llm->get_history_size();
}

void LLMAgent_Set_Overflow_Strategy(LLMAgent *llm, int strategy, float target_ratio, const char *summarize_prompt)
{
    constexpr int strategy_min = static_cast<int>(ContextOverflowStrategy::None);
    constexpr int strategy_max = static_cast<int>(ContextOverflowStrategy::Summarize);
    if (strategy < strategy_min || strategy > strategy_max)
    {
        std::cerr << "LLMAgent_Set_Overflow_Strategy: invalid strategy " << strategy << std::endl;
        return;
    }
    ContextOverflowStrategy s = static_cast<ContextOverflowStrategy>(strategy);

    std::string prompt = summarize_prompt ? summarize_prompt : "";
    if (prompt.empty())
        llm->set_overflow_strategy(s, target_ratio);
    else
        llm->set_overflow_strategy(s, target_ratio, prompt);
}

const char *LLMAgent_Get_Summary(LLMAgent *llm)
{
    return stringToCharArray(llm->get_summary());
}

void LLMAgent_Set_Summary(LLMAgent *llm, const char *summary)
{
    llm->set_summary(summary ? summary : "");
}
