#include "LlamaLib.h"
#include <iostream>
#include <thread>
#include <chrono>

static void streaming_callback(const char *c)
{
    std::cout << c << std::flush;
}

int main(int argc, char **argv)
{
    std::string server_url = "http://localhost";
    int server_port = 13333;

    // Create a remote client that connects to the server
    LLMClient llm_client(server_url, server_port);

    std::string PROMPT = "you are an artificial intelligence assistant\n\n--- user: Hello, how are you?\n--- assistant";

    std::cout << "----------------------- tokenize -----------------------" << std::endl;
    std::vector<int> tokens = llm_client.tokenize(PROMPT);
    std::cout << "tokens: ";
    for (int token : tokens)
    {
        std::cout << token << " ";
    }
    std::cout << std::endl;

    std::cout << std::endl
              << "----------------------- detokenize -----------------------" << std::endl;
    std::string detokenize_response = llm_client.detokenize(tokens);
    std::cout << "prompt: " << detokenize_response << std::endl;

    std::cout << std::endl
              << "----------------------- completion (streaming) -----------------------" << std::endl;
    std::cout << "response: ";
    llm_client.completion(PROMPT, static_cast<CharArrayFn>(streaming_callback));
    std::cout << std::endl;

    std::cout << std::endl
              << "----------------------- completion (no streaming) -----------------------" << std::endl;
    std::string completion_response = llm_client.completion(PROMPT);
    std::cout << "response: " << completion_response << std::endl;

    std::cout << std::endl
              << "----------------------- embeddings -----------------------" << std::endl;
    std::vector<float> embeddings = llm_client.embeddings(PROMPT);
    std::cout << "embeddings: ";
    size_t maxCount = std::min<size_t>(embeddings.size(), 10);
    for (size_t i = 0; i < maxCount; ++i)
    {
        std::cout << embeddings[i] << " ";
    }
    std::cout << "..." << std::endl;

    return 0;
}