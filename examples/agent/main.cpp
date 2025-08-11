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
    std::string user_message2 = "How are you today?";
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

    // Demonstrate manual message addition
    std::cout << std::endl
              << "----------------------- Manual Message Addition -----------------------" << std::endl;
    agent->add_user_message("This is a manually added user message");
    agent->add_assistant_message("This is a manually added assistant response");
    std::cout << "Added manual messages. New history size: " << agent->get_history_size() << std::endl;

    // Cleanup
    llm_service->stop();
    delete agent;
    delete llm_service;

    return 0;
}