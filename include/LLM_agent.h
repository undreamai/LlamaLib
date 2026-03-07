/// @file LLM_agent.h
/// @brief High-level conversational agent interface for LLMs
/// @ingroup llm
/// @details Provides a conversation-aware wrapper around LLM functionality with
/// chat history management

#pragma once

#include "LLM.h"
#include "LLM_client.h"

/// @brief Structure representing a single chat message
/// @details Encapsulates a conversation message with role and content,
/// providing JSON serialization and comparison functionality
struct UNDREAMAI_API ChatMessage
{
    std::string role;    ///< Message role (e.g., "user", "assistant", "system")
    std::string content; ///< Message content text

    /// @brief Default constructor
    ChatMessage() = default;

    /// @brief Parameterized constructor
    /// @param role_ Message role identifier
    /// @param content_ Message content text
    ChatMessage(const std::string &role_, const std::string &content_)
        : role(role_), content(content_) {}

    /// @brief Convert message to JSON representation
    /// @return JSON object with role and content fields
    /// @details Serializes the message for storage or transmission
    json to_json() const
    {
        return json{{"role", role}, {"content", content.empty() ? " " : content}};
    }

    /// @brief Create message from JSON representation
    /// @param j JSON object containing message data
    /// @return ChatMessage instance created from JSON
    /// @details Deserializes a message from JSON format
    /// @throws json::exception if required fields are missing
    static ChatMessage from_json(const json &j)
    {
        return ChatMessage(j.at("role").get<std::string>(), j.at("content").get<std::string>());
    }

    /// @brief Equality comparison operator
    /// @param other Another ChatMessage to compare with
    /// @return true if both role and content are identical, false otherwise
    bool operator==(const ChatMessage &other) const
    {
        return role == other.role && content == other.content;
    }
};

/// @brief Strategy to apply when the chat history would exceed the model's context window
enum class ContextOverflowStrategy
{
    None,       ///< No automatic handling — may crash if context is exceeded
    Truncate,   ///< Remove oldest messages (in pairs) from the front until history fits within target_context_ratio
    Summarize   ///< Summarise the full history (rolling chunks if needed), embed it in the system message, then truncate if still needed
};

const std::string SUMMARY_PROMPT=
"You are maintaining a concise working memory of an ongoing conversation."
""
"If an existing summary is provided, merge it with the new messages into a single updated summary."
"If no existing summary is provided, create a new summary from the messages."
""
"Rules:"
"- Preserve user goals, decisions made, constraints, preferences, open questions, and pending tasks."
"- Remove anything resolved, superseded, redundant, or purely conversational."
"- Keep only information relevant for future reasoning."
"- Avoid duplicating or rephrasing information unnecessarily."
"- Write in present tense where possible."
"- Keep under 200 words."
"- No bullet points. No preamble. Output only the summary text.";

/// @brief High-level conversational agent for LLM interactions
/// @details Provides a conversation-aware interface that manages chat history
/// and applies chat template formatting
class UNDREAMAI_API LLMAgent : public LLMLocal
{
public:
    const std::string USER_ROLE = "user";
    const std::string ASSISTANT_ROLE = "assistant";

    /// @brief Constructor for LLM agent
    /// @param llm Pointer to LLMLocal instance to wrap
    /// @param system_prompt Initial system prompt for conversation context
    /// @details Creates an agent that manages conversations with the specified LLM backend
    LLMAgent(LLMLocal *llm, const std::string &system_prompt = "");

    //=================================== LLM METHOD DELEGATES ===================================//
    /// @brief Tokenize input (override)
    /// @param data JSON object containing text to tokenize
    /// @return JSON string with token data
    std::string tokenize_json(const json &data) override { return llm->tokenize_json(data); }

    /// @brief Convert tokens back to text
    /// @param data JSON object containing token IDs
    /// @return JSON string containing detokenized text
    /// @details Pure virtual method for converting token sequences back to text
    std::string detokenize_json(const json &data) override { return llm->detokenize_json(data); }

    /// @brief Generate embeddings with HTTP response support
    /// @param data JSON object containing embedding request
    /// @return JSON string with embedding data
    /// @details Protected method used internally for server-based embedding generation
    std::string embeddings_json(const json &data) override { return llm->embeddings_json(data); }

    /// @brief Generate completion (delegate to wrapped LLM)
    /// @param data JSON completion request
    /// @param callback Optional streaming callback
    /// @param callbackWithJSON Whether callback uses JSON
    /// @return Generated completion
    std::string completion_json(const json &data, CharArrayFn callback = nullptr, bool callbackWithJSON = true) override { return llm->completion_json(data, callback, callbackWithJSON); }

    /// @brief Apply a chat template to message data
    /// @param data JSON object containing messages to format
    /// @return Formatted string with template applied
    /// @details Pure virtual method for applying chat templates to conversation data
    std::string apply_template_json(const json &data) override { return llm->apply_template_json(data); }

    /// @brief Manage slots with HTTP response support
    /// @param data JSON object with slot operation
    /// @return JSON response string
    /// @details Protected method used internally for server-based slot management
    std::string slot_json(const json &data) override { return llm->slot_json(data); }

    /// @brief Cancel request (delegate to wrapped LLM)
    /// @param data JSON cancellation request
    void cancel(int id_slot) override { return llm->cancel(id_slot); }

    /// @brief Get available slot (delegate to wrapped LLM)
    /// @return Available slot ID
    int get_next_available_slot() override { return llm->get_next_available_slot(); }

    /// @brief Get slot context size (delegate to wrapped LLM)
    /// @return Slot context size
    int get_slot_context_size() override { return llm->get_slot_context_size(); }

    //=================================== LLM METHOD DELEGATES ===================================//

    //=================================== Slot-aware method overrides ===================================//
    /// @brief Build completion JSON with agent's slot
    /// @param prompt Input prompt text
    /// @return JSON object for completion request
    /// @details Override that automatically uses the agent's assigned slot
    virtual json build_completion_json(const std::string &prompt) { return LLMLocal::build_completion_json(prompt, this->id_slot); }

    /// @brief Generate completion with agent's slot
    /// @param prompt Input prompt text
    /// @param callback Optional streaming callback
    /// @param return_response_json Whether to return JSON response
    /// @return Generated completion text or JSON
    /// @details Override that automatically uses the agent's assigned slot
    virtual std::string completion(const std::string &prompt, CharArrayFn callback = nullptr, bool return_response_json = false)
    {
        return LLMLocal::completion(prompt, callback, this->id_slot, return_response_json);
    }

    /// @brief Build slot operation JSON with agent's slot
    /// @param action Slot operation action ("save" or "restore")
    /// @param filepath File path for slot operation
    /// @return JSON object for slot operation
    /// @details Override that automatically uses the agent's assigned slot
    virtual json build_slot_json(const std::string &action, const std::string &filepath) { return LLMLocal::build_slot_json(this->id_slot, action, filepath); }

    /// @brief Save agent's slot state
    /// @param filepath Path to save slot state
    /// @return Operation result string
    /// @details Saves the agent's current processing state to file
    virtual std::string save_slot(const std::string &filepath) { return LLMLocal::save_slot(this->id_slot, filepath); }

    /// @brief Load agent's slot state
    /// @param filepath Path to load slot state from
    /// @return Operation result string
    /// @details Restores the agent's processing state from file
    virtual std::string load_slot(const std::string &filepath) { return LLMLocal::load_slot(this->id_slot, filepath); }

    /// @brief Cancel agent's current request
    /// @details Cancels any running request on the agent's slot
    virtual void cancel() { llm->cancel(this->id_slot); }
    //=================================== Slot-aware method overrides ===================================//

    /// @brief Get current processing slot ID
    /// @return Current slot ID
    /// @details Returns the slot ID used for this agent's operations
    inline int get_slot() { return id_slot; }

    /// @brief Set processing slot ID
    /// @param id_slot Slot ID to use for operations
    /// @details Assigns a specific slot for this agent's processing (not available for remote LLMClient)
    void set_slot(int id_slot);

    // Prompt configuration methods

    /// @brief Set system prompt
    /// @param system_prompt_ New system prompt text
    /// @details Sets the system prompt and clears conversation history
    void set_system_prompt(const std::string &system_prompt_) { system_prompt = system_prompt_; }

    /// @brief Get current system prompt
    /// @return Current system prompt string
    std::string get_system_prompt() const { return system_prompt; }

    /// @brief Set conversation history
    /// @param history_ JSON array of chat messages
    /// @details Replaces current conversation history with provided messages
    void set_history(const json &history_) { history = history_; }

    /// @brief Get conversation history
    /// @return JSON array containing conversation history
    /// @details Returns the complete conversation history as JSON
    json get_history() const { return history; }

    // History management methods

    /// @brief Add a user message to conversation history
    /// @param content User message content
    /// @details Convenience method for adding user messages
    void add_user_message(const std::string &content) { add_message(USER_ROLE, content); }

    /// @brief Add an assistant message to conversation history
    /// @param content Assistant message content
    /// @details Convenience method for adding assistant messages
    void add_assistant_message(const std::string &content) { add_message(ASSISTANT_ROLE, content); }

    /// @brief Clear all conversation history
    /// @details Removes all messages from the conversation history
    void clear_history();

    /// @brief Remove the last message from history
    /// @details Removes the most recently added message
    void remove_last_message();

    /// @brief Save conversation history to file
    /// @param filepath Path to save the history file
    /// @details Saves the current conversation history as JSON to the specified file
    void save_history(const std::string &filepath) const;

    /// @brief Load conversation history from file
    /// @param filepath Path to the history file to load
    /// @details Loads conversation history from a JSON file, replacing current history
    void load_history(const std::string &filepath);

    /// @brief Get number of messages in history
    /// @return Number of messages in conversation history
    /// @details Returns the count of messages currently stored in history
    size_t get_history_size() const { return history.size(); }

    // Context overflow management

    /// @brief Configure how the agent handles context overflow
    /// @param strategy The overflow strategy to use
    /// @param target_ratio Fraction of context to target after truncation (0.0–1.0, default 0.5)
    /// @param summarize_prompt Prompt used to ask the LLM to summarise the history
    void set_overflow_strategy(
        ContextOverflowStrategy strategy,
        float target_ratio = 0.5f,
        const std::string &summarize_prompt = SUMMARY_PROMPT
    )
    {
        overflow_strategy = strategy;
        target_context_ratio = target_ratio;
        this->summarize_prompt = summarize_prompt;
    }

    /// @brief Get the current overflow strategy
    ContextOverflowStrategy get_overflow_strategy() const { return overflow_strategy; }

    /// @brief Get the current summarize prompt
    std::string get_summarize_prompt() const { return summarize_prompt; }

    /// @brief Get the current rolling summary (empty if none has been generated yet)
    std::string get_summary() const { return summary; }

    /// @brief Set the rolling summary directly (e.g. after loading from file)
    void set_summary(const std::string &summary_) { summary = summary_; }

    // Chat functionality

    /// @brief Conduct a chat interaction
    /// @param user_prompt User's input message
    /// @param add_to_history Whether to add messages to conversation history
    /// @param callback Optional callback for streaming responses
    /// @param return_response_json Whether to return full JSON response
    /// @param debug_prompt Whether to display the complete prompt (default: false)
    /// @return Assistant's response text or JSON
    /// @details Main chat method that processes user input, applies conversation context,
    /// generates a response, and optionally updates conversation history
    std::string chat(const std::string &user_prompt, bool add_to_history = true, CharArrayFn callback = nullptr, bool return_response_json = false, bool debug_prompt = false);

protected:
    void set_n_keep();

    /// @brief Builds the history to send to the model including only the prompts
    /// @return JSON array: [system+summary]
    json build_system_history() const;

    /// @brief Build the full message list to send to the model
    /// @param user_prompt The current user message to append
    /// @param include_history Whether to include the chat history
    /// @return JSON array: [system+summary, ...history, user_prompt]
    json build_working_history(const std::string &user_prompt, bool include_history=true) const;

    /// @brief Handle context overflow using the configured strategy before a chat call
    /// @param user_prompt The user prompt string about to be sent
    /// @return true if history was modified
    bool handle_overflow(const std::string &user_prompt);

    /// @brief Remove oldest message pairs from the front until history fits within target_context_ratio
    void truncate_history(const std::string &user_prompt);

    /// @brief Summarise the entire history (chunking if needed), embed summary in system message, then truncate if still needed
    void summarize_history(const std::string &user_prompt);

    /// @brief Add a message to conversation history
    /// @param role Message role identifier
    /// @param content Message content text
    /// @details Appends a new message to the conversation history
    virtual void add_message(const std::string &role, const std::string &content);

private:
    LLMLocal *llm = nullptr;                  ///< Wrapped LLM instance
    int id_slot = -1;                         ///< Assigned processing slot ID
    std::string system_prompt = "";           ///< System prompt for conversation context
    std::string system_role = "system";       ///< Role identifier for system messages
    std::string summary = "";                 ///< Rolling summary embedded into the system message by Summarize strategy
    json history;                             ///< Conversation history as JSON array

    // Context overflow
    ContextOverflowStrategy overflow_strategy = ContextOverflowStrategy::Truncate;
    float target_context_ratio = 0.5f;        ///< Target fill ratio after truncation (0.0–1.0)
    std::string summarize_prompt = SUMMARY_PROMPT;
};

/// @ingroup c_api
/// @{

extern "C"
{
    /// @brief Construct LLMAgent (C API)
    /// @param llm LLMLocal instance to wrap
    /// @param system_prompt Initial system prompt (default: "")
    /// @return Pointer to new LLMAgent instance
    /// @details Creates a conversational agent with the specified configuration
    UNDREAMAI_API LLMAgent *LLMAgent_Construct(LLMLocal *llm, const char *system_prompt = "");

    /// @brief Set system prompt (C API)
    /// @param llm LLMAgent instance pointer
    /// @param system_prompt New system prompt string
    /// @details Setting system prompt clears conversation history
    UNDREAMAI_API void LLMAgent_Set_System_Prompt(LLMAgent *llm, const char *system_prompt);

    /// @brief Get system prompt (C API)
    /// @param llm LLMAgent instance pointer
    /// @return Current system prompt string
    UNDREAMAI_API const char *LLMAgent_Get_System_Prompt(LLMAgent *llm);

    /// @brief Set processing slot (C API)
    /// @param llm LLMAgent instance pointer
    /// @param slot_id Slot ID to assign
    UNDREAMAI_API void LLMAgent_Set_Slot(LLMAgent *llm, int slot_id);

    /// @brief Get processing slot (C API)
    /// @param llm LLMAgent instance pointer
    /// @return Current slot ID
    UNDREAMAI_API int LLMAgent_Get_Slot(LLMAgent *llm);

    /// @brief Conduct chat interaction (C API)
    /// @param llm LLMAgent instance pointer
    /// @param user_prompt User input message
    /// @param add_to_history Whether to save messages to history (default: true)
    /// @param callback Optional streaming callback function
    /// @param return_response_json Whether to return JSON response (default: false)
    /// @param debug_prompt Whether to display the complete prompt (default: false)
    /// @return Generated assistant response
    /// @details Main chat method for conversational interactions
    UNDREAMAI_API const char *LLMAgent_Chat(LLMAgent *llm, const char *user_prompt, bool add_to_history = true, CharArrayFn callback = nullptr, bool return_response_json = false, bool debug_prompt = false);

    /// @brief Clear conversation history (C API)
    /// @param llm LLMAgent instance pointer
    /// @details Removes all messages from conversation history
    UNDREAMAI_API void LLMAgent_Clear_History(LLMAgent *llm);

    /// @brief Get conversation history (C API)
    /// @param llm LLMAgent instance pointer
    /// @return JSON string containing conversation history
    UNDREAMAI_API const char *LLMAgent_Get_History(LLMAgent *llm);

    /// @brief Set conversation history (C API)
    /// @param llm LLMAgent instance pointer
    /// @param history_json JSON string containing conversation history
    /// @details Replaces current history with provided JSON data
    UNDREAMAI_API void LLMAgent_Set_History(LLMAgent *llm, const char *history_json);

    /// @brief Add user message to history (C API)
    /// @param llm LLMAgent instance pointer
    /// @param content Message content text
    /// @details Appends a new message to conversation history
    UNDREAMAI_API void LLMAgent_Add_User_Message(LLMAgent *llm, const char *content);

    /// @brief Add assistant message to history (C API)
    /// @param llm LLMAgent instance pointer
    /// @param content Message content text
    /// @details Appends a new message to conversation history
    UNDREAMAI_API void LLMAgent_Add_Assistant_Message(LLMAgent *llm, const char *content);

    /// @brief Remove last message from history (C API)
    /// @param llm LLMAgent instance pointer
    /// @details Removes the most recently added message from history
    UNDREAMAI_API void LLMAgent_Remove_Last_Message(LLMAgent *llm);

    /// @brief Save conversation history to file (C API)
    /// @param llm LLMAgent instance pointer
    /// @param filepath Path to save history file
    /// @details Saves conversation history as JSON to specified file
    UNDREAMAI_API void LLMAgent_Save_History(LLMAgent *llm, const char *filepath);

    /// @brief Load conversation history from file (C API)
    /// @param llm LLMAgent instance pointer
    /// @param filepath Path to history file to load
    /// @details Loads conversation history from JSON file
    UNDREAMAI_API void LLMAgent_Load_History(LLMAgent *llm, const char *filepath);

    /// @brief Get conversation history size (C API)
    /// @param llm LLMAgent instance pointer
    /// @return Number of messages in conversation history
    UNDREAMAI_API size_t LLMAgent_Get_History_Size(LLMAgent *llm);

    /// @brief Configure the context overflow strategy (C API)
    /// @param llm LLMAgent instance pointer
    /// @param strategy 0=None, 1=Truncate, 2=Summarize
    /// @param target_ratio Target fill ratio after truncation (0.0–1.0)
    /// @param summarize_prompt Prompt used for summarization (nullptr = use default)
    UNDREAMAI_API void LLMAgent_Set_Overflow_Strategy(LLMAgent *llm, int strategy, float target_ratio, const char *summarize_prompt);

    /// @brief Get the current rolling summary (C API)
    /// @param llm LLMAgent instance pointer
    /// @return Summary string, or empty string if none has been generated
    UNDREAMAI_API const char *LLMAgent_Get_Summary(LLMAgent *llm);

    /// @brief Set the rolling summary directly (C API)
    /// @param llm LLMAgent instance pointer
    /// @param summary Summary string to restore
    UNDREAMAI_API void LLMAgent_Set_Summary(LLMAgent *llm, const char *summary);
}

/// @}