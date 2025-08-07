#include "LlamaLib.h"
#include <iostream>

static void streaming_callback(const char *c)
{
    std::cout << c;
}

int main(int argc, char **argv)
{
    std::string model = "model.gguf";

    // Create the underlying LLM service
    LLMService *llm_service = new LLMService(model);
    llm_service->start();

    // Create an agent with a system prompt
    std::string system_prompt = "You are a helpful AI assistant. Be concise and friendly.";
    LLMAgent *agent = new LLMAgent(llm_service, system_prompt);

    std::cout << "=== LLM Agent Conversation Example ===" << std::endl;
    std::cout << "System prompt: " << agent->get_system_prompt() << std::endl;
    std::cout << "User role: " << agent->get_user_role() << std::endl;
    std::cout << "Assistant role: " << agent->get_assistant_role() << std::endl;
    std::cout << std::endl;

    // First conversation turn
    std::cout << "----------------------- First Turn -----------------------" << std::endl;
    std::string user_message1 = "Hello! What's your name?";
    std::cout << "User: " << user_message1 << std::endl;
    std::cout << "Assistant: ";
    std::string response1 = agent->chat(user_message1, true, static_cast<CharArrayFn>(streaming_callback));
    std::cout << std::endl;

    // Second conversation turn (maintains context)
    std::cout << std::endl
              << "----------------------- Second Turn -----------------------" << std::endl;
    std::string user_message2 = "What did I just ask you?";
    std::cout << "User: " << user_message2 << std::endl;
    std::cout << "Assistant: ";
    std::string response2 = agent->chat(user_message2, true, static_cast<CharArrayFn>(streaming_callback));
    std::cout << std::endl;

    // Show conversation history
    std::cout << std::endl
              << "----------------------- Conversation History -----------------------" << std::endl;
    std::cout << "History size: " << agent->get_history_size() << " messages" << std::endl;
    json history = agent->get_history();
    for (const auto &msg : history)
    {
        std::cout << msg["role"].get<std::string>() << ": " << msg["content"].get<std::string>() << std::endl;
    }

    // Save conversation history
    std::cout << std::endl
              << "----------------------- Save/Load History -----------------------" << std::endl;
    std::string history_file = "conversation_history.json";
    agent->save_history(history_file);
    std::cout << "History saved to: " << history_file << std::endl;

    // Clear history and reload
    agent->clear_history();
    std::cout << "History cleared. Size: " << agent->get_history_size() << std::endl;

    agent->load_history(history_file);
    std::cout << "History loaded. Size: " << agent->get_history_size() << std::endl;

    // Continue conversation after reload
    std::cout << std::endl
              << "----------------------- After Reload -----------------------" << std::endl;
    std::string user_message3 = "Can you summarize our conversation?";
    std::cout << "User: " << user_message3 << std::endl;
    std::cout << "Assistant: ";
    std::string response3 = agent->chat(user_message3, true, static_cast<CharArrayFn>(streaming_callback));
    std::cout << std::endl;
    std::cout << "New history size: " << agent->get_history_size() << std::endl;

    // Demonstrate manual message addition
    std::cout << std::endl
              << "----------------------- Manual Message Addition -----------------------" << std::endl;
    agent->add_user_message("This is a manually added user message");
    agent->add_assistant_message("This is a manually added assistant response");
    std::cout << "Added manual messages. New history size: " << agent->get_history_size() << std::endl;

    // Test slot operations (save/restore agent state)
    std::cout << std::endl
              << "----------------------- Slot Operations -----------------------" << std::endl;
    std::string slot_file = "agent_slot.bin";
    std::string save_result = agent->save_slot(slot_file);
    std::cout << "Slot save result: " << save_result << std::endl;

    std::string load_result = agent->load_slot(slot_file);
    std::cout << "Slot load result: " << load_result << std::endl;

    // Cleanup
    llm_service->stop();
    delete agent;
    delete llm_service;

    return 0;
}