#include <iostream>

#include "LlamaLib.h"

#define ASSERT(cond)                                           \
    do                                                         \
    {                                                          \
        if (!(cond))                                           \
        {                                                      \
            std::cerr << "Assertion failed: " << #cond << "\n" \
                      << "File: " << __FILE__ << "\n"          \
                      << "Line: " << __LINE__ << std::endl;    \
            std::abort();                                      \
        }                                                      \
    } while (false)

static void streaming_callback(const char *c)
{
    std::cout << c;
}

int main(int argc, char **argv)
{
    std::string model = "../model.gguf";

    LLMService *llm_service = new LLMService(model);
    ASSERT(llm_service->debug_implementation() == "runtime_detection");

    llm_service->start();
    std::string PROMPT =
        "you are an artificial intelligence assistant\n\n--- user: Hello, how are you?\n--- "
        "assistant";

    std::cout << "----------------------- tokenize -----------------------" << std::endl;
    std::vector<int> tokens = llm_service->tokenize(PROMPT);
    std::cout << "tokens: ";
    for (int token : tokens)
    {
        std::cout << token << " ";
    }
    std::cout << std::endl;

    std::cout << std::endl
              << "----------------------- detokenize -----------------------" << std::endl;
    std::string detokenize_response = llm_service->detokenize(tokens);
    std::cout << "prompt: " << detokenize_response << std::endl;

    std::cout << std::endl
              << "----------------------- completion (streaming) -----------------------"
              << std::endl;
    std::cout << "response: ";
    llm_service->set_completion_params({{"seed", 0}, {"n_predict", 30}});
    llm_service->completion(PROMPT, static_cast<CharArrayFn>(streaming_callback));
    std::cout << std::endl;

    std::cout << std::endl
              << "----------------------- completion (no streaming) -----------------------"
              << std::endl;
    std::string completion_response =
        llm_service->completion(PROMPT, static_cast<CharArrayFn>(streaming_callback));
    std::cout << "response: " << completion_response << std::endl;

    std::cout << std::endl
              << "----------------------- embeddings -----------------------" << std::endl;
    std::vector<float> embeddings = llm_service->embeddings(PROMPT);
    std::cout << "embeddings: ";
    for (int embedding : embeddings)
    {
        std::cout << embedding << " ";
    }
    std::cout << std::endl;

    // Agent
    PROMPT = "You are a helpful AI assistant. Be concise and friendly.";
    LLMAgent agent(llm_service, PROMPT);

    std::cout << "response: ";
    agent.chat("how are you?", true, static_cast<CharArrayFn>(streaming_callback));
    std::cout << std::endl;

    llm_service->stop();
    delete llm_service;

    return 0;
}
