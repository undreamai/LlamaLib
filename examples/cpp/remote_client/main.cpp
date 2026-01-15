#include "LlamaLib.h"
#include <iostream>

static std::string previous_text = "";
static void streaming_callback(const char *c)
{
    std::string current_text(c);
    // streaming gets the entire generated response up to now, print only the new text
    std::cout << current_text.substr(previous_text.length()) << std::flush;
    previous_text = current_text;
}

int main(int argc, char **argv)
{
    std::string server_url = "http://localhost";
    std::string PROMPT = "Hello, how are you?";
    int server_port = 13333;

    // Create a remote client that connects to the server
    std::cout << "*** Using client ***" << std::endl;
    LLMClient llm_client(server_url, server_port);
    std::cout << "----------------------- tokenize -----------------------" << std::endl;
    std::vector<int> tokens = llm_client.tokenize(PROMPT);
    std::cout << "tokens: ";
    for (int token : tokens) std::cout << token << " ";
    std::cout << std::endl << std::endl;

    // Create an agent that uses the remote client
    std::cout << "*** Using agent ***" << std::endl;
    std::string system_prompt = "You are a helpful AI assistant. Be concise and friendly.";
    LLMAgent agent(&llm_client, system_prompt);
    std::cout << "----------------------- completion (streaming) using agent -----------------------" << std::endl;
    std::cout << "User: " << PROMPT << std::endl;
    std::cout << "Assistant: ";
    agent.chat(PROMPT, true, static_cast<CharArrayFn>(streaming_callback));
    std::cout << std::endl;

    return 0;
}
