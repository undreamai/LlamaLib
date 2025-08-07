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

    // Create a local client that wraps the service
    LLMClient *llm_client = new LLMClient(llm_service);

    llm_service->start();
    std::string PROMPT = "you are an artificial intelligence assistant\n\n--- user: Hello, how are you?\n--- assistant";

    std::cout << "----------------------- tokenize -----------------------" << std::endl;
    std::vector<int> tokens = llm_client->tokenize(PROMPT);
    std::cout << "tokens: ";
    for (int token : tokens)
    {
        std::cout << token << " ";
    }
    std::cout << std::endl;

    std::cout << std::endl
              << "----------------------- detokenize -----------------------" << std::endl;
    std::string detokenize_response = llm_client->detokenize(tokens);
    std::cout << "prompt: " << detokenize_response << std::endl;

    std::cout << std::endl
              << "----------------------- completion (streaming) -----------------------" << std::endl;
    std::cout << "response: ";
    llm_client->completion(PROMPT, static_cast<CharArrayFn>(streaming_callback));
    std::cout << std::endl;

    std::cout << std::endl
              << "----------------------- completion (no streaming) -----------------------" << std::endl;
    std::string completion_response = llm_client->completion(PROMPT);
    std::cout << "response: " << completion_response << std::endl;

    std::cout << std::endl
              << "----------------------- embeddings -----------------------" << std::endl;
    std::vector<float> embeddings = llm_client->embeddings(PROMPT);
    std::cout << "embeddings: ";
    for (float embedding : embeddings)
    {
        std::cout << embedding << " ";
    }
    std::cout << std::endl;

    delete llm_client;
    delete llm_service;

    return 0;
}