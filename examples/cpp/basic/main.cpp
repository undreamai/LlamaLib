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
    std::string PROMPT = "The capital of";

    // create LLM
    LLMService* llm_service = LLMServiceBuilder().model("model.gguf").numGPULayers(10).build();
    // alternatively using the LLMService constructor:
    // LLMService* llm_service = new LLMService("model.gguf", 1, -1, 10);
    llm_service->start();

    // Optional: limit the amount of tokens that we can predict so that it doesn't produce text forever (some models do)
    llm_service->set_completion_params({{"n_predict", 20}});

    std::cout << "----------------------- tokenize -----------------------" << std::endl;
    std::vector<int> tokens = llm_service->tokenize(PROMPT);
    std::cout << "tokens: ";
    for (int token : tokens) std::cout << token << " ";
    std::cout << std::endl;

    std::cout << std::endl << "----------------------- detokenize -----------------------" << std::endl;
    std::string detokenize_response = llm_service->detokenize(tokens);
    std::cout << "prompt: " << detokenize_response << std::endl;

    std::cout << std::endl << "----------------------- completion (streaming) -----------------------" << std::endl;
    std::cout << "response: ";
    llm_service->completion(PROMPT, streaming_callback);
    std::cout << std::endl;

    std::cout << std::endl  << "----------------------- completion (no streaming) -----------------------" << std::endl;
    std::string completion_response = llm_service->completion(PROMPT);
    std::cout << "response: " << completion_response << std::endl << std::endl;

    delete llm_service;

    return 0;
}
