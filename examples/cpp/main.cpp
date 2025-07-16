#include "LlamaLib.h"
#include <iostream>

static void streaming_callback(const char* c)
{
    std::cout << c;
}

int main(int argc, char** argv) {
    std::string model = "../../tests/model.gguf";
    
    // use the following for selection of architecture on runtime according to the user system
    LLMRuntime* llm_service = new LLMRuntime(model);
    // use the following for mobile platforms or if building for a specific architecture
    // LLMService* llm_service = new LLMService(model);

    llm_service->start();
    std::string PROMPT = "you are an artificial intelligence assistant\n\n--- user: Hello, how are you?\n--- assistant";

    std::cout << "----------------------- tokenize -----------------------" << std::endl;
    std::vector<int> tokens = llm_service->tokenize(PROMPT);
    std::cout<<"tokens: ";
    for(int token: tokens) { std::cout<<token<<" "; }
    std::cout << std::endl;

    std::cout << std::endl << "----------------------- detokenize -----------------------" << std::endl;
    std::string detokenize_response = llm_service->detokenize(tokens);
    std::cout << "prompt: " << detokenize_response << std::endl;

    std::cout << std::endl << "----------------------- completion (streaming) -----------------------" << std::endl;
    std::cout << "response: ";
    llm_service->completion(PROMPT, 0, static_cast<CharArrayFn>(streaming_callback));
    std::cout << std::endl;

    std::cout << std::endl << "----------------------- completion (no streaming) -----------------------" << std::endl;
    std::string completion_response = llm_service->completion(PROMPT, 0, static_cast<CharArrayFn>(streaming_callback));
    std::cout << "response: " << completion_response << std::endl;
    
    std::cout << std::endl << "----------------------- embeddings -----------------------" << std::endl;
    std::vector<float> embeddings = llm_service->embeddings(PROMPT);
    std::cout<<"embeddings: ";
    for(int embedding: embeddings) { std::cout<<embedding<<" "; }
    std::cout << std::endl;

    llm_service->stop();
    delete llm_service;

    return 0;
}
